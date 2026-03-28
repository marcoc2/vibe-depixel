# Backlog de Experimentos de Super-Resolução

Este documento consolida ideias e possibilidades para melhorar o pipeline de upscaling e Super-Resolução (SR) do projeto `vibe-depixel`, desde melhorias na arquitetura atual até a adoção de modelos Estado-da-Arte (SOTA).

---

## 1. Melhorias na Arquitetura Atual (`EDSRLite`)
A arquitetura atual é baseada numa versão condensada do EDSR. Podemos aplicar as seguintes melhorias iterativas sem abandonar a base lógica já construída:

- [ ] **Aumentar Capacidade (Escalar a Rede):**
  - Aumentar o número de canais (`n_feats`) de `64` para `128` ou `256`.
  - Aumentar o número de blocos residuais (`n_resblocks`) de `16` para `32`.
- [ ] **Atualizar Funções de Ativação:**
  - Substituir o uso de `ReLU` nos blocos residuais por funções mais robustas como `LeakyReLU` ou `GELU` para preservar informações sutis e evitar "neurônios mortos".
- [ ] **Mecanismos de Atenção:**
  - Implementar mecanismos de *Channel Attention* (ex: RCAN) dentro da classe `ResBlock` para focar o aprendizado nos canais e características visuais mais informativos.
- [ ] **Ajustes na Dinâmica de Treinamento (`sr_train.py`):**
  - Alterar o agendador de taxa de aprendizado (`StepLR`) para abordagens mais avançadas, como `CosineAnnealingLR` ou `CosineAnnealingWarmRestarts`.
  - Habilitar Treinamento de Precisão Mista (FP16 / AMP) para permitir lotes (batch sizes) maiores e gastar menos VRAM.
  - Implementar *Adversarial Loss* (Discriminador / modelo GAN) somado à sua *Perceptual Loss* para gerar texturas mais realistas num estilo SRGAN.

---

## 2. Adoção de Modelos SOTA (State-of-the-Art)
O campo de SR avançou significativamente. Abaixo estão as alternativas de ponta recomendadas caso o EDSR já tenha atingido seu "teto" na recuperação de detalhes do projeto, com foco especial na adoção de Transformers.

### 2.1. Foco Principal: SwinIR
O modelo SwinIR baseia-se em *Swin Transformers* em vez de convoluções clássicas, o que permite à rede analisar correlações globais (pedaços distantes da imagem) e produzir delimitações muito mais precisas sem borrar texturas locais.

* **Requisitos de Hardware Estimados:**
  * **RAM Clássica (Sistema):** 16 GB a 32 GB.
  * **VRAM (Placa de Vídeo):** 
    * `~8 GB` para a versão *Lightweight* (tamanhos de lote limitados).
    * `12 GB a 16 GB` para a arquitetura *Classical* (treinamentos em parâmetros normais).
    * `24 GB+` para versões *Real-World* ou pesadas focadas em texturas extremas.

* **Estratégia Mínima Viável de Implementação:**
  * **Não reinventar a roda:** Evitar escrever estruturas de Transformer como o SwinIR do zero usando o PyTorch puro clássico.
  * **Framework Dedicado:** Transitar ou adaptar as execuções de treino em repositórios maduros como o **BasicSR** (criado pelos autores do Real-ESRGAN/SwinIR, que já lida com gargalos e data loaders complexos no formato de arquivos de configuração `.yml`).
  * **Transfer Learning:** Consumir pesos prontos empacotados pelo **Hugging Face (`transformers`)** para realizar fine-tunings em vez de treinar o modelo cegamente do zero absoluto (from scratch) com o dataset da aplicação.

### 2.2. Alternativas Paralelas (Backup Plan)
Caso a escalabilidade/memória RAM do SwinIR mostre-se um empecilho muito agressivo ou exija complexidade de infraestrutura desproporcional ao ganho prático:

- [ ] **Avançar para Real-ESRGAN / ESRGAN:** Adota blocos RRDB (Residual-in-Residual Dense Blocks) extremamente maduros que utilizam perdas de GAN super otimizadas. Perfeito como salto imediato pois limpa perfeitamente blur e artefatos gerando grão ou ruído fotográfico natural, com consumo convolucional ainda tradicional.
- [ ] **Difusão Latente (Stable Diffusion Upscalers / ControlNet Tile / LDSR):** Redes com IA puramente generativa. São maravilhosas para criar detalhes esteticamente fotorealistas porque a rede literalmente "redesenha" informações perdidas baseadas no que acha que havia naquele pixel. **Contra:** Imprevisível, extremamente morosa para inferência contínua/vídeo em tempo real e não retém uma correspondência fixa do pixel natural na fidelidade 100%. Recomendada apenas se o projeto puder tolerar "alucinações" suaves.
