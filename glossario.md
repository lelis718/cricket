# Glossário — PyTorch / LLM / Redes Neurais

> Termos recolhidos dos ficheiros `cricket-explicado.md` e `arquitetura.md`,
> organizados por ordem alfabética. Cada definição é curta e ligada ao contexto
> do modelo Cricket. Ideal para consultar enquanto lê a documentação.

---

## A

**AdamW**
Otimizador usado para treinar o modelo (com `learning_rate`, ex.: 3e-4). Atualiza
cada parâmetro com base no gradiente e num "momento" por parâmetro; o "W" refere-se
à correção do *weight decay* (uma variante moderna e estável do Adam).

**Amostragem (sampling)**
Escolha do próximo token a partir de uma distribuição de probabilidades (via
`torch.multinomial`), em vez de pegar sempre o mais provável. Combina-se com
temperatura, top-k e repetition penalty para controlar a aleatoriedade.

**Autoregressivo**
Modo de geração token a token: cada novo token é calculado a partir de tudo o que
já foi gerado e anexado à sequência, que serve de entrada para o passo seguinte.

**Aux loss / Perda auxiliar**
Perda extra que incentiva o MoE a usar os experts de forma equilibrada. No
Cricket é calculada a partir da fração de tokens e da importância de cada expert,
mas não influencia o treino porque `aux_loss_weight = 0`.

---

## B

**Backpropagation (backward)**
Algoritmo que percorre o grafo de operações do PyTorch ao contrário para calcular
o *gradiente* de cada parâmetro em relação à perda (`loss.backward()`).

**Batch**
Conjunto de exemplos processados em paralelo numa mesma passada (`batch_size = 32`).

**Bias**
Termo de deslocamento (o `b` de `y = x·Wᵀ + b`) numa camada linear. O dado extra
que a camada aprende para ajustar a origem da ativação.

**BPE — Byte-Pair Encoding**
Algoritmo de tokenização que começa com letras/símbolos e funde repetidamente os
pares mais frequentes em "pedaços" maiores, até atingir `vocab_size` (~4000 no
Cricket). É o tokenizador usado no projeto.

**Broadcast**
Mecanismo do PyTorch que estica automaticamente shapes pequenos para combinar com
shapes maiores numa operação (ex.: somar uma máscara `(3,)` a scores `(B, T, 3)`).

---

## C

**Cache**
Ficheiros que evitam repetir trabalho: `dataset_cache.txt` (textos descarregados)
e `tokenizer.json` (tokenizador BPE treinado). Se existirem, são apenas lidos.

**Cabeça de saída (lm_head)**
Camada linear que transforma o vetor final de cada token em *logits* sobre todo o
vocabulário. No Cricket partilha pesos com o embedding (*weight tying*).

**Checkpoint**
Ficheiro `.pt` que guarda o `state_dict` (os pesos) do melhor modelo encontrado
durante o treino. Usado depois na inferência.

**Clip de gradientes**
Limitador da magnitude dos gradientes (`clip_grad_norm_(..., 1.0)`), para evitar
que valores gigantes "explodam" o treino.

**Connection residual (skip connection)**
Ligação que soma a entrada de um bloco à sua saída (`x = residual + x`). Cada
bloco só precisa aprender a correção (Δ), não reconstruir tudo.

**Context embedding**
Embedding médio de uma sequência (prompt ou exemplo). No MoE, é usado pelo router
de prefetch para decidir os experts candidatos.

**Contiguous / view / transpose**
`view` rearranja a "interpretação" de um tensor sem mudar os dados; `transpose`
troca de ordem duas dimensões; `contiguous()` copia o tensor para memória contígua
(necessário antes de alguns `view` após um `transpose`).

**Cross-entropy**
Função de perda usada na previsão do próximo token: compara os logits do modelo
com o token real e devolve um número que diminui quando a previsão acerta com
confiança.

---

## D

**Dataset**
Conjunto de textos de treino. No projeto vem de cache em disco, de downloads
(*streaming*) ou do fallback manual (`dataset.txt`).

**DataLoader**
Utilitário do PyTorch que entrega os dados em *batches*, com opção de `shuffle`
(embaralhar a cada época no treino).

**Decodificação**
Passo que transforma IDs de volta em texto (`tokenizer.decode`), ignorando o
padding.

**Denso (arquitetura densa)**
Modelo com uma única FFN SwiGLU para todos os tokens (baseline, ~1,0M parâmetros).

**Device**
Dispositivo onde os cálculos correm: CPU (padrão do projeto) ou GPU (`--cuda`).

**Dropout**
Regularização que "desliga" aleatoriamente parte das ativações durante o treino,
forçando o modelo a não depender de caminhos específicos. Desligado em `model.eval()`.

---

## E

**Early stopping**
Interrupção do treino quando a perda de validação para de melhorar durante
`patience` épocas seguidas (ex.: 5), evitando sobreajuste e desperdício de tempo.

**Embedding (nn.Embedding)**
Tabela de consulta `(vocab_size, hidden_size)` que transforma o id de cada token
num vetor semântico (ex.: de 128 dimensões).

**Encode**
Conversão de texto em lista de IDs (`tokenizer.encode`).

**Época (epoch)**
Uma passada completa por todos os batches de treino. O Cricket usa 5 épocas.

**Experts (especialistas)**
No MoE, cada expert é uma FFN SwiGLU independente. No Cricket há 3 experts, dos
quais só `top_k` (1) é ativado por token.

---

## F

**Fallback**
Lista manual de frases (`dataset.txt`) usada quando não há rede nem cache. Nunca
é gravada no `dataset_cache.txt`.

**FFN — Feed-Forward Network**
Rede que processa cada token de forma isolada (após a atenção, que mistura os
tokens). No Cricket é o SwiGLU (denso) ou o `MoEWithPrefetch`.

**Forward (forward pass)**
Passagem para a frente: o dado entra, atravessa as camadas e sai como logits/loss.

---

## G

**Gate**
Mecanismo que decide "quanto passa". No SwiGLU, a projeção `w1` com SiLU faz o
papel de portão por dimensão; no MoE, o *gate token-a-token* pontua os experts
para cada token.

**Gather (recolher)**
Operação de indexação avançada que "apanha" elementos/tokens indicados por
índices. No MoE, `x[idx]` recolhe só os tokens que usam um dado expert para
passá-los em lote pelo expert.

**GQA — Grouped Query Attention**
Atenção em que o número de heads de Key/Value (`num_kv_heads`) é menor que o de
heads de Query (`num_heads`). Cada head de KV é repetido para servir um grupo de
queries, poupando memória com qualidade quase igual.

**Grafo de operações**
Estrutura criada pelo PyTorch no forward que registra todas as operações;
`loss.backward()` percorre-o ao contrário para calcular gradientes.

**Gradiente**
Vetor que indica, para cada parâmetro, como a perda varia com ele — a "direção"
em que o otimizador atualiza os pesos.

**Greedy (decoding)**
Escolha sempre do token mais provável (`argmax`), sem aleatoriedade. Equivale a
temperatura 0.

---

## H

**Head (cabeça de atenção)**
Subespaço independente de atenção. O Cricket divide `hidden_size` por `num_heads`
(128 / 4 = 32 dimensões por head).

**Head dim (head_dim)**
Dimensão de cada head de atenção (`hidden_size / num_heads`). RoPE e o factor de
escala (1/√head_dim) dependem dele.

**Hidden size**
Dimensão do vetor que representa cada token (128 no Cricket).

---

## I

**Indexação avançada (advanced indexing)**
Selecionar partes de um tensor com índices/máscaras em vez de slicing simples.
Ver **Gather** e **Scatter-add**; usada no forward dos experts do MoE para
recolher e devolver tokens (ex.: `x[idx]`, `final_out[idx] += ...`).

**Inferência**
Uso do modelo treinado para gerar texto (modo `infer`, loop de chat).

**Iteration**
Um passo de treino sobre um batch (forward + backward + update).

---

## K

**Key (K)**
Em atenção: "rótulo" de cada token — com o que ele se identifica para ser
encontrado pelas queries. Gerado por `wk`.

**KV cache**
Guarda Keys/Values passadas na inferência (aqui implícito). Ter menos heads de KV
(GQA) reduz este cache.

---

## L

**LayerNorm**
Normalização clássica que centraliza (subtrai média) e escala. O `RMSNorm` é uma
versão mais leve que apenas escala.

**Learning rate**
Tamanho do passo com que o otimizador atualiza os pesos a cada iteração (3e-4).

**LLM — Large Language Model**
Modelo que recebe uma sequência de tokens e devolve, para cada posição, uma
distribuição de probabilidade sobre o próximo token — sendo capaz de gerar texto.

**Logits**
"Placares" brutos (sem softmax) que o `lm_head` produz para cada token do
vocabulário. Podem ser negativos; o softmax converte-os em probabilidades.

**Loss (perda)**
Número que mede o erro do modelo. Desce durante o treino através do otimizador.

---

## M

**Máscara causal**
Triângulo superior com `-inf` aplicado aos scores de atenção, impedindo que um
token "veja" posições futuras. É o que torna a geração autoregressiva correta.

**Máscara de candidatos**
Em MoE: 0 para os experts pré-selecionados e `-inf` para os restantes; somada aos
scores do gate, invalida os não-candidatos.

**MoE — Mixture of Experts**
Arquitetura que substitui a FFN única por vários especialistas e roteia cada token
para os escolhidos. No Cricket, ~1,8M parâmetros, mas só 1/3 ativos por token.

**ModuleList**
Lista de módulos do PyTorch que registra automaticamente os parâmetros internos
(usada para as `num_layers` camadas).

**Multinomial**
`torch.multinomial` — amostra índices a partir das probabilidades dadas.

---

## N

**nn.Linear / nn.Parameter**
`nn.Linear` é uma camada densa `y = x·Wᵀ + b`. `nn.Parameter` é qualquer tensor
registado como treinável, encontrado por `model.parameters()`.

**No-grad (torch.no_grad)**
Contexto que desativa o grafo de gradações — poupa memória/CPU na inferência e
validação.

**Normalização**
Reescala vetores para magnitude saudável, estabilizando o treino (RMSNorm).

---

## P

**Padding (`<PAD>`)**
Preenchimento das sequências curtas até `max_seq_len` com o token id 0. Na perda,
essas posições são ignoradas (`ignore_index=0`).

**Perplexidade**
Métrica de qualidade de um modelo de linguagem (exponencial da perda média):
quanto menor, melhor.

**Parâmetros**
Números internos ajustáveis do modelo (pesos). Contados por `model.parameters()`.

**Prefetch por requisição**
Primeiro nível do roteamento MoE: a partir do embedding médio do prompt, seleciona
de antemão os `num_candidates` experts que podem responder (máscara guardada).

**Pre-normalização**
Colocar a normalização antes do sub-bloco (atenção/FFN) em vez de depois,
mantendo o caminho residual limpo.

**Pré-processamento**
Etapas antes do treino: limpeza, truncamento e padding dos textos.

---

## Q

**Query (Q)**
Em atenção: "pergunta" de cada token — o que ele está procurando entre os outros
tokens. Gerado por `wq`.

---

## R

**Register_buffer**
Forma de guardar tensores no módulo que acompanham o modelo mas não são
treináveis (ex.: tabelas de RoPE e máscara causal).

**Repeat_interleave**
Copia/trai cada elemento repetidamente ao longo de um eixo — usado no GQA para
duplicar heads de KV e servir múltiplas heads de query.

**Repetition penalty**
Na inferência: divide os logits dos tokens recentes (últimos 10 passos) por um
factor (>1), tornando-os menos prováveis e reduzindo loops ("gato gato gato...").

**RMSNorm**
Normalização por raiz quadrada média (`rsqrt(mean(x²)+eps)`), leve e usada em
modelos modernos como pré-normalização.

**RoPE — Rotary Positional Embedding**
Codifica posição rodando os vetores de query/key por um ângulo proporcional à
posição, com frequências de "velocidades" diferentes por dimensão. A distância
relativa entre tokens fica embutida na própria atenção.

**Router de Prefetch**
Camada linear que pontua os experts a partir do embedding médio do prompt
(`prefetch_router`).

**Roteamento**
Decisão de qual expert processa cada token (token-a-token) e quais ficam candidatos
(prefetch por requisição).

---

## S

**Scatter-add (espalhar somando)**
Espalhar resultados de volta nas posições de onde foram recolhidos, somando ao
que já lá está. No MoE, `final_out[idx] += weight_e[idx].unsqueeze(-1) * out`
devolve a saída de cada expert (ponderada) à posição original, acumulando a
contribuição de todos os experts.

**Seed (semente)**
Valor fixado (`seed = 42`) para tornar a execução reproduzível: mesmos números
aleatórios, mesmos resultados.

**Shape**
Forma/dimensões de um tensor (ex.: `[B, 128]`, `[B, T, ...]`). A última dimensão
é normalmente o "vetor de features".

**SiLU / Sigmoid**
Ativações: `sigmoid` comprime para [0,1]; SiLU (`x·sigmoid(x)`) permite valores
negativos, deixando o gate amplificar, atenuar ou inverter.

**Softmax**
Converte um vetor de scores em probabilidades que somam 1.

**Split (divisão)**
Separação dos dados: 80% treino / 20% validação (`train_test_split`).

**State_dict**
Dicionário com todos os pesos do modelo, guardado/recarregado em checkpoints.

**Streaming**
Modo de carregar datasets HuggingFace por lotes, sem baixar tudo à memória.

**SwiGLU**
FFN "com portão": `w2( SiLU(w1(x)) × w3(x) )`. Camadas gate/up/down; é a base dos
experts do MoE e do modelo denso.

---

## T

**Temperatura**
Parâmetro de inferência que divide os logits antes do softmax: baixa → previsível,
alta → aleatório. Zero é greedy.

**Tensor**
Array N-dimensional do PyTorch que rastreia operações para cálculo automático de
gradientes.

**Token**
Pedaço de texto convertido num índice inteiro pelo tokenizador.

**Tokenizador**
Componente que converte texto em IDs (e vice-versa). No Cricket é BPE (~4000 tokens).

**Top-k (no MoE)**
Número de experts ativados por token (`top_k = 1`).

**Top-k sampling**
Na inferência: manter apenas os `top_k` tokens mais prováveis (os demais vão para
`-inf`) antes de amostrar.

**torch.compile**
Otimização opcional do PyTorch que compila o modelo para execução mais rápida
(`--compile`); pode não estar disponível em todos os CPUs.

**Train / Eval (modos)**
`model.train()`: ativa dropout e o grafo de gradações; `model.eval()`: desativa-os
para inferência/validação.

**Transformer**
Arquitetura base de blocos com atenção + FFN e conexões residuais.

**Truncamento**
Corte de sequências acima de `max_seq_len` (64 tokens no Cricket).

---

## V

**Validação**
Conjunto de 20% dos textos (nunca vistos durante o treino) usado para medir
generalização e decidir *early stopping*.

**Vetorização**
Substituir loops sobre elementos por operações sobre tensores inteiros (mais
rápidas e padronizadas). Ex.: a combinação dos experts do MoE usa máscara ×
pesos + gather/scatter-add em vez de um loop token a token.

**Value (V)**
Em atenção: "conteúdo" entregue pelo token se for escolhido. Gerado por `wv`.

**Vocabulário (vocab_size)**
Tamanho real do conjunto de tokens do tokenizador (~4000), definido
dinamicamente em `CONFIG["vocab_size"]`.

---

## W

**Weight decay**
Termo de regularização no otimizador que "encolhe" pesos grandes (parte do AdamW).

**Weight tying**
Partilha dos pesos entre `lm_head` e o embedding (`lm_head.weight = embedding.weight`):
"token→vetor" e "vetor→token" usam o mesmo dicionário. Menos parâmetros, melhor
em modelos pequenos.

---

*Consultas: [`cricket-explicado.md`](cricket-explicado.md) e [`arquitetura.md`](arquitetura.md).*