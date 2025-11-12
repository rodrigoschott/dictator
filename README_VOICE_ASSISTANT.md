# Dictator Voice Assistant - Modo Automático 🎙️🤖

Conversas bidirecionais **totalmente automáticas** em voz com Claude - **100% LOCAL** (exceto Claude Code).

## 🚀 Arquitetura

```
┌────────────────────────────────────────────────────────────────┐
│                    100% LOCAL PROCESSING                        │
│                                                                  │
│  Você fala → Whisper STT → temp/voice_input.json               │
│                               ↓                                  │
│                         [MCP Server]                             │
│                               ↓                                  │
├───────────────────────────────────────────────────────────────

─┤
│                    CLAUDE CODE (Cloud)                           │
│                                                                  │
│  Custom Agent (@voice-assistant) em loop contínuo:              │
│    1. get_pending_voice_input() via MCP                        │
│    2. Processa pergunta                                         │
│    3. send_voice_response() via MCP                             │
│    4. GOTO 1 (loop automático)                                  │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                    100% LOCAL PROCESSING                         │
│                                                                  │
│                    temp/voice_output.json                        │
│                               ↓                                  │
│                  [Voice Response Handler]                        │
│                         (monitora)                               │
│                               ↓                                  │
│                         Kokoro TTS                               │
│                               ↓                                  │
│                      🔊 Você ouve                                │
└──────────────────────────────────────────────────────────────────┘
```

## ✅ Componentes 100% Locais

- **STT**: faster-whisper (RTX 5080 GPU)
- **TTS**: kokoro-onnx (RTX 5080 GPU)
- **MCP Server**: dictator MCP tools
- **Voice Response Handler**: monitora respostas (thread local)
- **Arquivos temp/**: comunicação via JSON local

## ☁️ Componente Cloud (Permitido)

- **Claude Code**: Executa agent customizado em loop que chama MCP tools

## 📋 Configuração

### 1. Registrar MCP Server no Claude Code

**Já foi feito automaticamente!** Verifique:

```bash
# No diretório D:\Dev\py\Dictator
cat .claude.json | grep dictator
```

Deve mostrar:
```json
"dictator": {
  "type": "stdio",
  "command": "poetry",
  "args": ["run", "python", "src/dictator/mcp_server.py"]
}
```

### 2. Ativar Voice Assistant Mode

**No Dictator:**
- Clique direito no ícone da bandeja
- Marque ✅ **"Voice Assistant Mode"**

Isso ativa:
- Conversation Manager (local)
- Voice Response Handler (local, monitora respostas)
- MCP Server fica disponível para Claude Code

### 3. Iniciar Claude Code com Custom Agent

**No Claude Code** (dentro do diretório D:\Dev\py\Dictator):

```
@voice-assistant
```

Ou use o slash command:

```
/voice-assistant
```

O Claude entrará automaticamente em **loop contínuo** executando:
```python
while True:
    input = get_pending_voice_input()  # MCP tool
    if input:
        response = process(input)
        send_voice_response(response)  # MCP tool
    wait(2s)
```

## 🎯 Como Usar

### Início de Conversa

1. **Ative Voice Assistant Mode** no Dictator (tray icon)
2. **No Claude Code**, execute: `@voice-assistant`
3. Claude responderá: "Voice Assistant Mode activated, monitoring for input..."
4. **Fale** pressionando o botão do mouse
5. **Ouça a resposta** automaticamente!

### Exemplo de Fluxo Completo

```
[Você]
  Pressiona botão do mouse
  Fala: "Qual é a capital da França?"

[Dictator - LOCAL]
  ✅ Whisper transcreve → "Qual é a capital da França?"
  ✅ Salva em temp/voice_input.json

[Claude Code - CLOUD]
  ✅ Agent chama get_pending_voice_input() via MCP
  ✅ Recebe: "Qual é a capital da França?"
  ✅ Processa resposta
  ✅ Chama send_voice_response("A capital...") via MCP

[Dictator - LOCAL]
  ✅ Response Handler detecta temp/voice_output.json
  ✅ Kokoro TTS fala: 🔊 "A capital da França é Paris..."

[Claude Code - CLOUD]
  ✅ Agent continua loop automaticamente
  ✅ Aguardando próximo input...
```

## 🛠️ MCP Tools Disponíveis

### Tools (Chamados pelo Claude Code)

| Tool | Descrição | Local/Cloud |
|------|-----------|-------------|
| `get_pending_voice_input()` | Lê nova entrada de voz | ✅ Local |
| `send_voice_response(text)` | Envia resposta para TTS | ✅ Local |
| `get_conversation_history(n)` | Obtém histórico | ✅ Local |
| `add_to_conversation_history(role, content)` | Adiciona ao histórico | ✅ Local |
| `clear_conversation_history()` | Limpa histórico | ✅ Local |

### Resources (Leitura pelo Claude Code)

| Resource | Descrição | Local/Cloud |
|----------|-----------|-------------|
| `config://dictator` | Config.yaml | ✅ Local |
| `logs://dictator` | Logs recentes | ✅ Local |

## ⚙️ Customização do Agent

### Editar Comportamento do Agent

Edite `.claude/agents/voice-assistant.md`:

```markdown
---
name: Voice Assistant
description: Maintains continuous voice conversation loop
model: sonnet  # ou opus para respostas melhores
---

# Customize aqui:
- Personalidade do assistente
- Estilo de resposta
- Idiomas suportados
- Comportamento do loop
```

### Ajustar System Prompt

Em `config.yaml`:

```yaml
mcp:
  claude:
    system_prompt: "You are a Python programming expert assistant..."
```

### Modificar Loop do Agent

Edite `.claude/agents/voice-assistant.md` e ajuste:

```python
# Intervalo de checagem
wait(2s)  # Altere para 1s para respostas mais rápidas

# Tamanho do histórico
get_conversation_history(max_entries=20)
```

## 📊 Performance

| Componente | Latência | Onde Roda |
|------------|----------|-----------|
| Whisper STT | ~500ms | 🖥️ Local GPU |
| MCP get_pending | ~10ms | 🖥️ Local |
| Claude processa | ~1-2s | ☁️ Cloud |
| MCP send_response | ~10ms | 🖥️ Local |
| Kokoro TTS | ~100ms | 🖥️ Local GPU |
| **TOTAL** | **~2-3s** | - |

## 🐛 Troubleshooting

### "MCP server not available"

**Causa**: Claude Code não está no diretório correto

**Solução**:
```bash
cd D:\Dev\py\Dictator
claude
# Agora execute: @voice-assistant
```

### "Agent não entra em loop"

**Causa**: Agent não foi ativado corretamente

**Solução**:
1. Certifique-se de estar em `D:\Dev\py\Dictator`
2. Execute: `@voice-assistant` (com @)
3. Aguarde mensagem de confirmação

### "Respostas não são faladas"

**Verificações**:

1. **Response Handler rodando?**
   ```bash
   tail -f logs/dictator.log | grep VoiceResponseHandler
   # Deve mostrar: "Voice response handler started"
   ```

2. **TTS habilitado?**
   ```yaml
   # config.yaml
   tts:
     enabled: true
   ```

3. **Voice Assistant Mode ativo?**
   - Clique direito no tray → ✅ Voice Assistant Mode

### "Agent para de funcionar"

**Causa**: Erro no loop ou timeout

**Solução**:
1. Verifique logs do Claude Code
2. Re-execute: `@voice-assistant`
3. Se persistir, reinicie Claude Code

## 🔧 Configuração Avançada

### Usar Opus para Respostas Mais Elaboradas

Edite `.claude/agents/voice-assistant.md`:

```markdown
---
model: opus  # Mudança aqui
---
```

### Múltiplos Idiomas Simultâneos

```markdown
## Response Guidelines
**Language**: Detect and respond in user's language (PT, EN, ES, etc.)
```

### Respostas Mais Longas/Curtas

```markdown
**Conciseness**:
- Quick answers: 1 sentence
- Detailed explanations: 3-5 sentences (when asked)
```

## 📝 Arquivos de Comunicação Local

Todos locais em `temp/`:

### `voice_input.json`
```json
{
  "text": "Qual é a capital da França?",
  "timestamp": "2025-11-11T16:30:00",
  "processed": false
}
```

### `voice_output.json`
```json
{
  "text": "A capital da França é Paris.",
  "timestamp": "2025-11-11T16:30:02",
  "spoken": false
}
```

### `conversation_history.json`
```json
[
  {"role": "user", "content": "...", "timestamp": "..."},
  {"role": "assistant", "content": "...", "timestamp": "..."}
]
```

## 🎨 Estados Visuais

| Estado | Ícone/Cor | Descrição |
|--------|-----------|-----------|
| Idle | ⚪ Branco | Aguardando |
| Recording | 🔴 Vermelho | Gravando voz |
| Processing | 🟡 Amarelo | STT transcrevendo |
| Speaking | 🟢 Verde | TTS falando |

## 🔐 Privacidade e Segurança

### 100% Local:
- ✅ STT (Whisper)
- ✅ TTS (Kokoro)
- ✅ Arquivos temporários
- ✅ MCP Server
- ✅ Response Handler
- ✅ Histórico de conversa

### Cloud (Apenas Claude Code):
- ☁️ Processamento de linguagem (Claude API via Claude Code)

**Nenhum dado é enviado para terceiros além da API do Claude via Claude Code!**

## 🚀 Comandos Rápidos

```bash
# Ativar modo
# 1. Dictator tray → Voice Assistant Mode ✅

# 2. Claude Code
cd D:\Dev\py\Dictator
@voice-assistant

# Testar
# Pressione botão do mouse e fale!

# Parar
# Claude Code: diga "exit voice mode" ou Ctrl+C
# Dictator: desmarque Voice Assistant Mode
```

## 📚 Diferenças vs Implementação Anterior

| Aspecto | Versão Anterior (API) | Versão Atual (MCP) |
|---------|----------------------|-------------------|
| API calls diretas | ❌ Anthropic SDK | ✅ Nenhuma |
| Processamento local | 50% | 95% |
| Cloud dependency | Claude API | Claude Code apenas |
| Automático | ✅ Sim | ✅ Sim (via agent) |
| Setup | Complexo (API key) | Simples (só ativar) |

## 🎉 Pronto para Usar!

1. ✅ MCP Server registrado
2. ✅ Custom agent criado
3. ✅ Response handler local
4. ✅ TTS configurado

**Comece agora:**
```
Dictator: Ative Voice Assistant Mode
Claude Code: @voice-assistant
Você: Fale qualquer coisa!
```

---

**Dúvidas?** Consulte `logs/dictator.log` ou `logs/mcp_server.log`

**100% Local. 100% Privado. 100% Automático.** 🎙️🤖
