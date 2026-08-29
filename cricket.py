# -*- coding: utf-8 -*-
"""
#################################################################
  MODELO CRICKET - LLM didático (Denso vs MoE com Prefetch)
#################################################################

Objetivo
--------
O Cricket é um laboratório de estudo em PyTorch que implementa duas
arquiteturas de linguagem e permite compará-las no mesmo hardware:

  * Denso  : FFN SwiGLU única (baseline)        -> ~1,0M parâmetros
  * MoE    : experts SwiGLU + roteamento em 2 níveis (prefetch por
             requisição + gate token-a-token)   -> ~1,8M parâmetros

A secção de cada bloco é referenciada ao longo do código com um
número (§2.1, §2.3, §3, §4) apontando para a documentação técnica
em `arquitetura.md`. Esta organização foi pensada para estudo:
os importantes detalhes do roteamento MoE ficam EXPLICITAMENTE
escritos num loop simples, em vez de serem escondidos em operações
tensor eficientes (mas opacas).

Uso via linha de comando
------------------------
  python cricket.py train dense     # treina o modelo denso
  python cricket.py train moe       # treina o modelo MoE
  python cricket.py infer dense     # chat com o modelo denso treinado
  python cricket.py infer moe       # chat com o modelo MoE treinado

Opções adicionais:
  --cuda                usa GPU se disponível (padrão: CPU)
  --compile             ativa torch.compile (opcional, pode falhar na CPU)
  --max-new-tokens N    tokens máximos por resposta no chat (padrão: 40)
  --temperature T       0 = greedy | >0 = amostragem  (padrão: 0.8)
  --repetition-penalty P >1 penaliza repetições        (padrão: 1.2)
  --top-k K             >0 amostra apenas dos K mais prováveis (padrão: 0)

Fluxo de execução (funções)
---------------------------
  main()
    ├─ train  mode ──→ load_texts() -> BPETokenizer -> datasets/loaders
    │                   -> CricketLM -> train_model() -> plot_history()
    └─ infer  mode ──→ BPETokenizer (cache) -> CricketLM (checkpoint)
                        -> run_chat() -> generate_text()  [loop interactivo]
"""

# ============================================================
# 1. IMPORTS
# ============================================================
import os
import math
import random
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from tokenizers import Tokenizer                     # tokenizador BPE
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Suprime warnings (muitos são do tokenizers/datasets e inofensivos)
warnings.filterwarnings('ignore')


# ============================================================
# 2. CONFIGURAÇÃO GLOBAL
#    CONFIG segue exatamente a configuração recomendada da
#    documentação (§6, para 24 GB RAM). `vocab_size` é definido
#    dinamicamente depois do treino do tokenizador.
# ============================================================
CONFIG = {
    "hidden_size": 128,        # dimensão dos embeddings
    "num_layers": 2,           # número de blocos transformer
    "num_heads": 4,            # heads de query (hidden_size divisível por num_heads)
    "num_kv_heads": 2,         # heads de key/value (GQA, metade de num_heads)
    "ffn_hidden": 512,         # 4 * hidden_size (dimensão interna da FFN)
    "max_seq_len": 64,         # comprimento máximo da sequência
    "batch_size": 32,          # activações em memória (24 GB RAM)
    "dropout": 0.1,
    "learning_rate": 3e-4,
    "epochs": 20,
    "aux_loss_weight": 0,      # 0 => perda auxiliar computada mas não aplicada
    "num_experts": 3,          # MoE: número de especialistas SwiGLU
    "top_k": 1,                # MoE: nº de experts activos por token
    "num_candidates": 2,       # MoE: nº de experts pré-seleccionados pelo prefetch
    "seed": 42,
    "vocab_size": None,        # <- preenchido dinamicamente após o tokenizador
}

# Caminhos de cache (downloads/treinos da 1.ª execução)
DATASET_CACHE = "dataset_cache.txt"   # textos descarregados de fontes externas
DATASET_FALLBACK = "dataset.txt"      # lista manual de frases de fallback
TOKENIZER_CACHE = "tokenizer.json"    # tokenizador BPE treinado

# Constantes de recolha de dados
NUM_TEXTS = 5000              # nº máximo de textos a descarregar
TEXT_MIN_LEN = 100            # comprimento mínimo do texto recolhido
TEXT_MAX_LEN = 1000           # comprimento máximo do texto recolhido


# ============================================================
# 3. UTILITÁRIOS (seed, device e contagem de parâmetros)
# ============================================================
def set_seed(seed):
    """Torna a execução reproduzível fixando as sementes aleatórias."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(force_cpu=True):
    """
    Escolhe o dispositivo de execução.

    O projeto foi concebido para correr em CPU (documentação §1), por
    isso por omissão `force_cpu=True`. Com `--cuda`, tenta a GPU quando
    disponível e cai para CPU caso contrário.
    """
    if force_cpu or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device('cuda')


def count_parameters(model):
    """Conta todos os parâmetros treináveis do modelo."""
    return sum(p.numel() for p in model.parameters())


def checkpoint_path(arch):
    """Nome do ficheiro de checkpoint correspondente à arquitectura."""
    return f"cricket_model_{arch}_best.pt"


def load_checkpoint(model, path, device):
    """
    Carrega o estado do modelo de um checkpoint com compatibilidade.

    Os checkpoints antigos foram gravados por `torch.compile`, que
    guarda as chaves com o prefixo `_orig_mod.`. Aqui esse prefixo é
    removido para que o modelo (não compilado) carregue normalmente.
    """
    state = torch.load(path, map_location=device)
    if any(k.startswith("_orig_mod.") for k in state):
        print(f"  Checkpoint antigo detectado (prefixo `_orig_mod.`); a remover prefixo.")
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)


# ============================================================
# 4. DADOS
#    §3 da documentação.
#    (a) download de fontes externas -> cache em disco;
#    (b) BPETokenizer: BPE treinado sobre os textos, com cache;
#    (c) TextDataset: pad/truncado para `max_seq_len`.
# ============================================================
def _load_texts_from_file(path):
    """Lê uma lista de textos de um ficheiro (uma frase por linha)."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _collect(iterable, limit, min_len=TEXT_MIN_LEN, max_len=TEXT_MAX_LEN):
    """
    Recolhe textos de um iterável (streaming do `datasets`), filtrando
    por comprimento, até atingir `limit`.
    """
    out = []
    for example in iterable:
        text = " ".join(example["text"].split())  # limpeza básica (colapsa espaços)
        if min_len < len(text) < max_len:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def load_texts():
    """
    Carrega os textos de treino com a seguinte precedência:

      1. Se `dataset_cache.txt` existir  -> lê do disco (sem downloads);
      2. Senão, tenta descarregar, por ordem, mc4-pt, CrawlPT e brwac
         (modo streaming), guardando o resultado em `dataset_cache.txt`;
      3. Se nada funcionar, usa a lista manual de `dataset.txt`.
         IMPORTANTE: o fallback NUNCA é gravado em `dataset_cache.txt`,
         assim uma execução futura com rede volta a tentar os downloads.
    """
    texts = []

    # ---- 1. Cache em disco (sem downloads) ----
    if os.path.exists(DATASET_CACHE):
        texts = _load_texts_from_file(DATASET_CACHE)
        print(f"  Dataset carregado do cache: {DATASET_CACHE} ({len(texts)} textos)")

    # ---- 2. Download de fontes externas (uma única vez) ----
    if not texts:
        try:
            from datasets import load_dataset
        except ImportError:
            load_dataset = None
            print("  Aviso: biblioteca 'datasets' não disponível.")

        sources = [
            ("MC4-PT", "eduagarcia/mc4-pt", "train"),
            ("CrawlPT", "eduagarcia/CrawlPT", "train"),
            ("brwac", "dominguesm/brwac", "train"),
        ]

        if load_dataset is not None:
            for name, dataset_id, split in sources:
                if len(texts) >= NUM_TEXTS:
                    break
                try:
                    print(f"  A descarregar {name}...")
                    ds = load_dataset(dataset_id, split=split, streaming=True)
                    novos = _collect(ds, limit=NUM_TEXTS - len(texts))
                    texts += novos
                    print(f"  {name}: +{len(novos)} textos (total {len(texts)}).")
                    del ds
                except Exception as e:
                    print(f"  Aviso: {name} falhou: {e}")

        # Guarda APENAS texto descarregado da internet em cache
        if texts:
            with open(DATASET_CACHE, "w", encoding="utf-8") as f:
                f.write("\n".join(texts) + "\n")
            print(f"  Dataset descarregado e guardado em cache: {DATASET_CACHE}")

    # ---- 3. Fallback manual (nunca gravado em cache) ----
    if not texts and os.path.exists(DATASET_FALLBACK):
        texts = _load_texts_from_file(DATASET_FALLBACK)
        print(f"  Lista de fallback carregada de {DATASET_FALLBACK} ({len(texts)} frases).")

    if not texts:
        raise RuntimeError(
            f"Sem dados: nem {DATASET_CACHE}, nem fontes externas, nem {DATASET_FALLBACK} "
            "estão disponíveis. Verifique a ligação à rede."
        )

    return texts


class BPETokenizer:
    """
    Tokenizador BPE treinado sobre os textos do dataset (§2.1 / §3).

    Se `cache_path` existir, o tokenizador é carregado do disco (sem
    re-treinar). Caso contrário, é treinado dos `texts` e guardado.
    O tamanho final do vocabulário (que pode ser ligeiramente maior que
    `vocab_size` pedido) fica disponível em `.vocab_size`.
    """
    def __init__(self, texts, vocab_size=4000, max_len=128, cache_path=None):
        self.vocab_size = vocab_size
        self.max_len = max_len

        if cache_path and os.path.exists(cache_path):
            # --- Carrega tokenizador do cache ---
            self.tokenizer = Tokenizer.from_file(cache_path)
            self.tokenizer.enable_padding(pad_id=0, pad_token="<PAD>")
            self.tokenizer.enable_truncation(max_length=max_len)
            print(f"[BPE Tokenizer] Carregado do cache: {cache_path}")
        else:
            # --- Inicializa e treina o tokenizador BPE ---
            self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
            self.tokenizer.pre_tokenizer = Whitespace()

            # Trainer com tokens especiais (id 0 = PAD)
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=["<PAD>", "<UNK>"],
                min_frequency=2,           # tokens com <2 ocorrências são ignorados
            )
            self.tokenizer.train_from_iterator(texts, trainer=trainer)

            # Padding e truncamento automáticos
            self.tokenizer.enable_padding(pad_id=0, pad_token="<PAD>")
            self.tokenizer.enable_truncation(max_length=max_len)

            # Guarda em cache para não repetir o treino nas próximas execuções
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                self.tokenizer.save(cache_path)
                print(f"[BPE Tokenizer] Treinado e guardado em cache: {cache_path}")

        # IDs dos tokens especiais (usados nas operações de geração)
        self.pad_token_id = self.tokenizer.token_to_id("<PAD>")
        self.unk_token_id = self.tokenizer.token_to_id("<UNK>")

        # Tamanho real do vocabulário
        self.vocab_size = self.tokenizer.get_vocab_size()
        print(f"[BPE Tokenizer] Vocabulário com {self.vocab_size} tokens.")

    def encode(self, text):
        """Converte texto em lista de IDs (sem padding aqui)."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        """Converte IDs de volta para texto, ignorando tokens de padding."""
        filtered = [i for i in ids if i != self.pad_token_id]
        return self.tokenizer.decode(filtered)


class TextDataset(Dataset):
    """
    Dataset PyTorch: cada exemplo é uma sequência de IDs com
    `max_len` posições, truncada e preenchida com `<PAD>` (=0).
    Quando usado como alvo (`targets=batch`), o próprio texto é a
    supervisão: o modelo prevê o próximo token (shift dentro da perda).
    """
    def __init__(self, texts, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        for t in texts:
            ids = tokenizer.encode(t)
            if len(ids) > max_len:
                ids = ids[:max_len]
            ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
            self.data.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_data_loaders(texts, tokenizer, config):
    """Divide os textos (80/20) e devolve (train_loader, val_loader)."""
    train_texts, val_texts = train_test_split(
        texts, test_size=0.2, random_state=config["seed"]
    )

    train_dataset = TextDataset(train_texts, tokenizer, config["max_seq_len"])
    val_dataset = TextDataset(val_texts, tokenizer, config["max_seq_len"])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    print(f"  Treino: {len(train_dataset)} frases | Validação: {len(val_dataset)} frases")
    return train_loader, val_loader


# ============================================================
# 5. BLOCOS BASE  (§2.1)
#    RMSNorm, RoPE, GQA (Grouped Query Attention) e SwiGLU.
# ============================================================
class RMSNorm(nn.Module):
    """Normalização por RMS (§2.1) - usada na pré-normalização.
    Normaliza pelo valor quadrático médio em vez da variância,
    sem parâmetros de deslocamento, apenas um factor escalar `weight`."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    RoPE - Rotary Positional Embedding (§2.1).
    Roda as queries/keys por um ângulo proporcional à posição do token,
    codificando a ordem da sequência sem parâmetros adicionais.
    `rotate_half` implementa a rotação 2D por pares de dimensões.
    """
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    GQA - Grouped Query Attention (§2.1).
    Tem `num_heads` heads de query mas apenas `num_kv_heads` heads de
    key/value (cada head KV serve um grupo de heads Q), reduzindo a
    memória do cache KV em troca de pouca perda de qualidade.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = self.hidden_size // self.num_heads

        assert self.head_dim % 2 == 0, "RoPE exige head_dim par."

        self.wq = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.wk = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.wv = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)
        self.dropout = nn.Dropout(config["dropout"])

        # Cache das frequências RoPE (pré-calculadas para a sequência máxima)
        self.register_buffer("cos", torch.zeros(config["max_seq_len"], self.head_dim // 2))
        self.register_buffer("sin", torch.zeros(config["max_seq_len"], self.head_dim // 2))
        self._build_rope_cache(config["max_seq_len"])

        # Máscara causal: True nas posições futuras (que NÃO podem ser vistas)
        causal_mask = torch.triu(
            torch.ones(config["max_seq_len"], config["max_seq_len"], dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _build_rope_cache(self, seq_len):
        # Frequências geométricas (10000^(-2i/d)) e posições t=0..seq_len-1
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos()[None, None, :, :]
        self.sin = emb.sin()[None, None, :, :]

    def forward(self, x):
        B, T, _ = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Separa em heads: (B, T, heads, head_dim) -> (B, heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Aplica RoPE nas queries/keys (operação dependente da posição)
        cos = self.cos[:, :, :T, :self.head_dim]
        sin = self.sin[:, :, :T, :self.head_dim]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: expande as heads KV para igualar as heads Q (cada KV serve um grupo)
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # Scores de atenção com factor de escala 1/sqrt(head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Aplica a máscara causal (posições futuras -> -inf)
        mask = self.causal_mask[:T, :T]  # (T, T)
        attn = attn.masked_fill(mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    """
    FFN com activação SwiGLU (§2.1 / §2.2).
    Três projecções: gate (w1) e up (w3) são multiplicadas com
    a activação SiLU no gate, e o resultado passa por down (w2).
    Decompõe a FFN densa de 2 camadas na formulação "gated".
    """
    def __init__(self, config):
        super().__init__()
        h = config["hidden_size"]
        f = config["ffn_hidden"]
        self.w1 = nn.Linear(h, f, bias=False)  # Gate
        self.w2 = nn.Linear(f, h, bias=False)  # Down
        self.w3 = nn.Linear(h, f, bias=False)  # Up

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


# ============================================================
# 6. MoE COM PREFETCH POR REQUISIÇÃO  (§2.3)
# ============================================================
class MoEWithPrefetch(nn.Module):
    """
    FFN que substitui o SwiGLU denso por `num_experts` especialistas e
    roteia cada token em DOIS níveis:

      Nível 1 - Prefetch por requisição:
        `prefetch_router` linear mapeia o *embedding médio do prompt*
        para scores por expert e pré-selecciona `num_candidates`.
      Nível 2 - Gate token-a-token:
        `gate` linear calcula scores por token; a máscara do nível 1
        limita a escolha aos candidatos; o top-k selecciona os experts
        finais (ex.: 1) cujas saídas são combinadas por softmax.

    Além da saída `final_out`, devolve uma `aux_loss` de balanceamento
    que penaliza o uso desequilibrado dos experts (fração de tokens x
    importância ponderada uniforme). O peso dessa perda é controlado
    por `CONFIG["aux_loss_weight"]`.
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config["num_experts"]
        self.top_k = config["top_k"]
        self.num_candidates = config["num_candidates"]
        self.hidden_size = config["hidden_size"]

        # Especialistas (cada um é uma FFN SwiGLU completa)
        self.experts = nn.ModuleList([SwiGLU(config) for _ in range(self.num_experts)])

        # Nível 1: router de prefetch (por requisição)
        self.prefetch_router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Nível 2: gate token-a-token
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Máscara de candidatos do prefetch (estado EPÉMERO de inferência:
        # por isso é um atributo Python e não um buffer persistido no disco).
        self.candidate_mask = None

    def prefetch(self, context_embedding):
        """
        Nível 1: escolhe os `num_candidates` experts com base no embedding
        do prompt (`context_embedding`, dimensão H) e guarda uma máscara.
        Chamado uma vez por requisição, antes da geração (§4).
        """
        with torch.no_grad():
            scores = self.prefetch_router(context_embedding)
            _, top_m_idx = torch.topk(scores, k=self.num_candidates)
            # Máscara: 0.0 para os candidatos, -inf para os restantes.
            # Somar esta máscara aos scores do gate invalida os não-candidatos.
            mask = torch.full((self.num_experts,), float('-inf'), device=scores.device)
            mask[top_m_idx] = 0.0
            self.candidate_mask = mask
            return top_m_idx

    def forward(self, x, context_embedding=None):
        B, T, _ = x.shape

        # Nível 2: gate calcula scores para cada token (site por expert)
        scores = self.gate(x)  # (B, T, E)

        # --- Escolha da máscara de prefetch ---
        if context_embedding is not None:
            # Treino/validação: máscara dinâmica por batch, derivada do
            # embedding médio de cada exemplo, mantendo treino e validação
            # coerentes (o roteamento é sempre pré-filtrado pelos candidatos).
            pref_scores = self.prefetch_router(context_embedding)  # (B, E)
            _, top_m_idx = torch.topk(pref_scores, k=self.num_candidates, dim=-1)
            mask = torch.full((B, 1, self.num_experts), float('-inf'), device=x.device)
            mask.scatter_(
                dim=-1,
                index=top_m_idx.unsqueeze(1),
                src=torch.zeros_like(pref_scores.unsqueeze(1)),
            )
            masked_scores = scores + mask
        elif not self.training and self.candidate_mask is not None:
            # Inferência: usa a máscara pré-calculada em `prefetch()` (§4)
            masked_scores = scores + self.candidate_mask.view(1, 1, -1)
        else:
            # Fallback teórico (sem prefetch) -> roteamento puro por token
            masked_scores = scores

        # Top-K: escolhe os `top_k` melhores experts DENTRO dos candidatos
        top_k_weights, top_k_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)
        top_k_weights = torch.softmax(top_k_weights, dim=-1)

        # --- Perda auxiliar (balanceamento de utilização, §2.3) ---
        # Importância: soma dos pesos top-k atribuídos a cada expert
        importance = top_k_weights.sum(dim=[0, 1, 2])
        # Fração de tokens enviada a cada expert (via contagem de índices)
        idx_flat = top_k_indices.view(-1)
        frac = torch.bincount(idx_flat, minlength=self.num_experts).float()
        frac = frac / (B * T * self.top_k)
        importance = importance / (B * T)
        # Produto import x fração é menor quando a distribuição é uniforme
        aux_loss = (frac * importance).sum() * self.num_experts

        # --- Forward dos experts com combinação ponderada ---
        # FEITO DE FORMA EXPLÍCITA (loop) por motivos didácticos: mostra
        # exactamente como cada token soma a contribuição dos seus experts.
        final_out = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = (top_k_indices == e).any(dim=-1)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)          # tokens roteados p/ expert e
            inp = x[idx]
            out = self.experts[e](inp)
            for i, (b, t) in enumerate(zip(idx[0], idx[1])):
                pos = (top_k_indices[b, t] == e).nonzero(as_tuple=True)[0]
                if len(pos) > 0:
                    weight = top_k_weights[b, t, pos[0]]
                    final_out[b, t] += weight * out[i]

        return final_out, aux_loss


# ============================================================
# 7. BLOCO TRANSFORMER E MODELO CRICKET  (§2.1 / §2.3)
# ============================================================
class TransformerBlock(nn.Module):
    """
    Bloco com pré-normalização: Atenção GQA + FFN (densa ou MoE),
    ambos com norm+residual. A variante escolhida é definida por
    `use_moe`. Se MoE, a FFN devolve também a perda auxiliar.
    """
    def __init__(self, config, use_moe=True):
        super().__init__()
        self.use_moe = use_moe
        self.attention = GroupedQueryAttention(config)
        if use_moe:
            self.ffn = MoEWithPrefetch(config)   # §2.3
        else:
            self.ffn = SwiGLU(config)            # §2.2 (densa)
        self.norm1 = RMSNorm(config["hidden_size"])
        self.norm2 = RMSNorm(config["hidden_size"])

    def forward(self, x, context_embedding=None):
        # --- Atenção (pré-norm + residual) ---
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = residual + x

        # --- FFN (pré-norm + residual) ---
        residual = x
        x = self.norm2(x)
        if self.use_moe:
            x, aux_loss = self.ffn(x, context_embedding)
        else:
            x = self.ffn(x)
            aux_loss = 0.0
        x = residual + x
        return x, aux_loss


class CricketLM(nn.Module):
    """
    Modelo de linguagem completo:

      embeddings + N blocos Transformer + RMSNorm final + cabeça LM linear
      (com weight tying: a cabeça partilha os pesos do embedding §2.1).

    A perda é cross-entropy sobre o próximo token (ignorando padding),
    somada à perda auxiliar MoE ponderada por `aux_loss_weight`.
    """
    def __init__(self, config, use_moe=True):
        super().__init__()
        self.config = config
        self.use_moe = use_moe
        self.token_embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([
            TransformerBlock(config, use_moe) for _ in range(config["num_layers"])
        ])
        self.norm = RMSNorm(config["hidden_size"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying

    def forward(self, input_ids, targets=None, context_embedding=None):
        x = self.token_embedding(input_ids)  # (B, T, H)

        # Contexto do prefetch para o MoE:
        #  - Treino/validação (targets dado): embedding médio de cada exemplo
        #    do batch, criado dinamicamente por batch;
        #  - Inferência (sem targets): deixa None para usar a máscara de
        #    prefetch pré-calculada em `prefetch_all_layers` (§4).
        if context_embedding is None and self.use_moe and targets is not None:
            context_embedding = x.mean(dim=1)  # (B, H)

        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x, context_embedding)
            total_aux_loss += aux_loss

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Prevê o próximo token: logits[..., :-1] contra targets[..., 1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # ignora <PAD>=0
            ce_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss = ce_loss + self.config["aux_loss_weight"] * total_aux_loss

        return logits, loss

    def prefetch_all_layers(self, context_embedding):
        """
        Executa o prefetch (§2.3) em TODAS as camadas MoE, uma vez por
        requisição, antes da geração. Modelos densos não têm prefetch.
        """
        if not self.use_moe:
            print("  (modelo denso não tem prefetch)")
            return
        for layer in self.layers:
            if isinstance(layer.ffn, MoEWithPrefetch):
                layer.ffn.prefetch(context_embedding)


# ============================================================
# 8. TREINO  (§3)
# ============================================================
def train_model(model, train_loader, val_loader, config, device, checkpoint_path,
                patience=5):
    """
    Treina com AdamW, clip de gradientes a 1.0, validação por época e
    early stopping: para se a validação não melhorar por `patience`
    épocas. Guarda o melhor checkpoint (por val loss) em `checkpoint_path`.
    """
    print("\n🏋️  INICIANDO TREINO...")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config["epochs"]):
        # --- Treino ---
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            _, loss = model(batch, targets=batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validação ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, loss = model(batch, targets=batch)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        print(f"  Época {epoch+1:2d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # --- Early stopping + guardar o melhor modelo ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Melhor modelo guardado em '{checkpoint_path}' (val: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping na época {epoch+1} "
                      f"(sem melhoria por {patience} épocas).")
                break

    # Recarrega o melhor modelo encontrado
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path, device)
        print("Treino concluído. Melhor modelo recarregado.")
    return model, history


def plot_history(history):
    """Plota a curva de aprendizagem (train vs val loss por época)."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Treino')
    plt.plot(history['val_loss'], label='Validação')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Curva de Aprendizado')
    plt.grid(True)
    plt.show()


# ============================================================
# 9. INFERÊNCIA + CHAT  (§4)
# ============================================================
def generate_text(model, tokenizer, prompt, max_new_tokens=30, temperature=0.8,
                  repetition_penalty=1.2, top_k=0, device='cpu', verbose=True):
    """
    Gera texto a partir de um prompt.

    Funções de amostragem (§4):
      * temperature      : 0 = greedy (determinístico) | >0 = amostragem;
      * repetition_penalty: >1 penaliza tokens dos últimos 10 passos;
      * top_k            : >0 mantém apenas os K tokens mais prováveis.

    Para modelos MoE, o prefetch por requisição é calculado ANTES do
    loop: o embedding médio do prompt passa por `prefetch_all_layers`,
    fixando os experts candidatos de todas as camadas.

    Devolve:
        generated_ids (torch.Tensor) : prompt + tokens gerados
        full_text  (str)             : decodificação completa
        new_text   (str)             : apenas a parte gerada (novos tokens)
    """
    model.eval()
    model.to(device)

    if verbose:
        print(f"\n🔮 GERANDO TEXTO...")
        print(f"📝 Prompt: '{prompt}'")
        print(f"   Temperatura: {temperature}, Repetition penalty: {repetition_penalty}")

    # Tokeniza o prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # --- Prefetch por requisição para MoE (§4) ---
    if model.use_moe:
        with torch.no_grad():
            emb = model.token_embedding(input_tensor)       # (1, T, H)
            context_emb = emb.mean(dim=1).squeeze(0)        # (H,)
            model.prefetch_all_layers(context_emb)

    generated = input_tensor.clone()

    # --- Loop de geração ---
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(generated)                    # (1, seq_len, vocab)
            next_logits = logits[:, -1, :]                  # (1, vocab)

            # Repetition penalty: divide os logits dos tokens recentes
            if repetition_penalty is not None and repetition_penalty > 1.0:
                last_tokens = generated[0, -10:].tolist()
                for token_id in set(last_tokens):
                    if token_id != tokenizer.pad_token_id:
                        next_logits[0, token_id] /= repetition_penalty

            # Top-K sampling: zera tudo que não estiver nos K mais prováveis
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            # Amostragem com temperatura ou greedy
            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Concatena o novo token
            generated = torch.cat([generated, next_token], dim=-1)

            # Imprime o token em tempo real (streaming)
            if verbose:
                print(tokenizer.decode([next_token.item()]), end='', flush=True)

            # Paragem quando o modelo gera <PAD>
            if next_token.item() == tokenizer.pad_token_id:
                break

    if verbose:
        print()  # quebra de linha final do streaming

    full_text = tokenizer.decode(generated[0].tolist())
    new_text = tokenizer.decode(generated[0, input_tensor.shape[1]:].tolist())
    return generated, full_text, new_text


def run_chat(model, tokenizer, device, max_new_tokens=40, temperature=0.8,
             repetition_penalty=1.2, top_k=0):
    """
    Loop interactivo de chat: espera um input no terminal e devolve a
    resposta do modelo no output. Escreva 'exit', 'sair' ou 'q' para terminar.
    """
    print("\n💬 CHAT ATIVADO - escreva a sua mensagem (ou 'exit' para sair)")
    print(f"   (max_new_tokens={max_new_tokens}, temp={temperature}, "
          f"rep_penalty={repetition_penalty}, top_k={top_k})\n")

    while True:
        try:
            prompt = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaindo do chat...")
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "sair", "q"}:
            break

        print("Cricket: ", end='', flush=True)
        generate_text(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            device=device,
            verbose=True,        # streaming token-a-token do output
        )


# ============================================================
# 10. EXECUÇÃO PRINCIPAL (interface de linha de comando)
# ============================================================
def parse_args(argv=None):
    """Lê os argumentos da linha de comando (ex.: `python cricket.py infer moe`)."""
    p = argparse.ArgumentParser(
        description="Cricket - LLM didático (Denso vs MoE com Prefetch).",
        usage="python cricket.py <train|infer> <dense|moe> [opções]",
    )
    p.add_argument("mode", choices=["train", "infer"], help="train ou infer")
    p.add_argument("arch", choices=["dense", "moe"], help="denso ou MoE")
    p.add_argument("--cuda", action="store_true",
                   help="usar GPU se disponível (padrão: CPU)")
    p.add_argument("--compile", action="store_true",
                   help="ativar torch.compile (opcional; pode falhar na CPU)")
    p.add_argument("--max-new-tokens", type=int, default=40,
                   help="tokens máximos por resposta no chat (padrão: 40)")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="0 = greedy | >0 = amostragem (padrão: 0.8)")
    p.add_argument("--repetition-penalty", type=float, default=1.2,
                   help=">1 penaliza repetições (padrão: 1.2)")
    p.add_argument("--top-k", type=int, default=0,
                   help=">0 amostra apenas dos K mais prováveis (padrão: 0)")
    return p.parse_args(argv)


def main(argv=None):
    """Ponto de entrada: prepara dados/tokenizador/modelo e despacha o modo."""
    args = parse_args(argv)

    # Semente global para reprodutibilidade
    set_seed(CONFIG["seed"])

    # Dispositivo (CPU por omissão; `--cuda` habilita GPU quando disponível)
    device = get_device(force_cpu=not args.cuda)

    print("=" * 60)
    print(f" MODELO CRICKET - Modo: {args.mode.upper()} | Arquitetura: {args.arch.upper()}")
    print(f" Dispositivo: {device}")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1) Textos + tokenizador (com cache em disco)
    #    - train : precisa dos textos para treinar/carregar o tokenizer
    #    - infer : carrega apenas o tokenizer do cache (sem downloads);
    #              se o cache não existir, treina-o a partir dos textos.
    # ----------------------------------------------------------
    texts = None
    if args.mode == "train" or not os.path.exists(TOKENIZER_CACHE):
        texts = load_texts()

    tokenizer = BPETokenizer(
        texts if texts is not None else [],
        vocab_size=4000,
        max_len=CONFIG["max_seq_len"],
        cache_path=TOKENIZER_CACHE,
    )
    CONFIG["vocab_size"] = tokenizer.vocab_size       # §2.1: vocab real do BPE
    print(f"  Vocabulário configurado no modelo: {CONFIG['vocab_size']} tokens\n")

    # ----------------------------------------------------------
    # 2) Modelo
    # ----------------------------------------------------------
    use_moe = (args.arch == "moe")
    model = CricketLM(CONFIG, use_moe=use_moe)

    # Compilação opcional (torna o treino mais rápido em alguns backends,
    # mas não está disponível em todos os ambientes/CPUs)
    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"  Aviso: torch.compile não disponível ({e}). A continuar sem compilar.")

    total_params = count_parameters(model)
    print(f"  Parâmetros totais: {total_params:,}")
    if use_moe:
        # Estimativa simplificada de parâmetros activos por token (§2.3)
        active = CONFIG["top_k"] * CONFIG["num_layers"] * (
            CONFIG["ffn_hidden"] * CONFIG["hidden_size"] * 3) * 2
        print(f"  Parâmetros activos por token (estimado): {active:,}")
    print()

    # ----------------------------------------------------------
    # 3) Modo de execução
    # ----------------------------------------------------------
    if args.mode == "train":
        train_loader, val_loader = build_data_loaders(
            texts, tokenizer, CONFIG
        )
        ckpt = checkpoint_path(args.arch)
        model, history = train_model(
            model, train_loader, val_loader, CONFIG, device,
            checkpoint_path=ckpt, patience=5,
        )
        print(f"\n Treino concluído. Use: python cricket.py infer {args.arch}")
        plot_history(history)

    else:  # infer
        ckpt = checkpoint_path(args.arch)
        if not os.path.exists(ckpt):
            print(f" Checkpoint '{ckpt}' não encontrado. Execute o treino primeiro:")
            print(f"   python cricket.py train {args.arch}")
            return

        load_checkpoint(model, ckpt, device)
        model.to(device)
        print(f"  Modelo carregado de '{ckpt}'.\n")

        run_chat(
            model, tokenizer, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()