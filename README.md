# LOOP (Layered Operational Ontology Pipeline)

> **A cognitive architecture for AI agents that actually remembers.**

LOOP is a novel approach to AI memory management inspired by cognitive science. Instead of treating every conversation turn equally, LOOP implements a three-tier memory hierarchy with Bayesian importance tracking that lets agents decide what matters and what to forget.

## Why LOOP?

Most AI agents treat memory as an afterthought: dump everything into a vector database and hope retrieval works. LOOP takes a different approach:

- **Cognitive realism**: Three-tier memory (working → associative → long-term) mirrors human memory systems
- **Bayesian reasoning**: Dynamic importance scoring means agents learn what matters in context
- **Graceful degradation**: Limited working memory forces prioritization, not just infinite context windows
- **Composable architecture**: Use the full stack or just the pieces you need

LOOP isn't trying to replace RAG, it's the missing layer between your retrieval system and your LLM that makes conversations feel coherent over time.

## What Makes It Different

| Traditional Approach | LOOP |
|---------------------|------|
| Flat vector store | Three-tier memory hierarchy |
| Static embeddings | Bayesian importance tracking |
| Retrieve everything | Selective consolidation |
| Context window limits | Graceful memory decay |
| One-size-fits-all | Configurable persona system |

## Installation

```bash
pip install -e .
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

## Core Components

### Memory Systems
- **`WorkingMemory`**: Slot-based buffer with decay and Bayesian importance tracking. The "attention" layer that holds what's immediately relevant.
- **`AssociativeMemory`**: Episodic storage with similarity-based recall. Your agent's short-term memory for recent context.
- **`LongTermMemory`**: Persistent knowledge consolidation to disk. Where important facts go to live forever (or until you delete them).

### Control & Coordination
- **`Controller`**: Manages attention and energy across memory tiers. Decides what to retrieve, when to consolidate, and how to balance exploration vs. exploitation.
- **`MemoryConsolidator`**: Automatic promotion pipeline. Watches working memory and promotes high-importance content through associative → long-term tiers.

### Persona & Generation
- **`PersonaProfile`**: Configurable tone, response style, and behavioral guidelines. Load from a text file and your agent adopts a personality.
- **`PersonaCoordinator`**: Bridges memory systems and persona enforcement. Builds prompts that respect both retrieved context and persona constraints.

### API Integration
- **`GroqEngine`** / **`OpenAIEngine`**: Drop-in adapters for Groq (fast, free) and OpenAI APIs. Swap providers without changing your code.

## Quick Start

### Basic Usage

```python
from loop_core import WorkingMemory, WorkingMemoryConfig
import torch

# Create a working memory with 6 slots
wm = WorkingMemory(WorkingMemoryConfig(slots=6, use_bayesian=True))

# Add some context
embedding = torch.randn(384)  # Your embedding vector
metadata = {"text": "User prefers dark mode", "importance_hint": 0.8}
wm.update(embedding, metadata)

# Check what's in memory
for slot in wm.summary():
    print(f"Slot {slot['index']}: {slot['metadata']['text']} (importance: {slot['importance']:.2f})")
```

### Full Stack Example

```python
from loop_core import (
    WorkingMemory, WorkingMemoryConfig,
    AssociativeMemory, AssociativeMemoryConfig,
    LongTermMemory, LongTermMemoryConfig,
    Controller, ControllerConfig,
    MemoryConsolidator,
    load_persona_profile,
    get_engine
)

# Initialize memory hierarchy
wm = WorkingMemory(WorkingMemoryConfig(slots=6))
am = AssociativeMemory(AssociativeMemoryConfig(capacity=512))
ltm = LongTermMemory(LongTermMemoryConfig(storage_path="memory/knowledge.jsonl"))
controller = Controller(wm, am, ControllerConfig())

# Set up consolidation pipeline
consolidator = MemoryConsolidator(
    working_memory=wm,
    associative_memory=am,
    long_term_memory=ltm,
    encoder=your_sentence_transformer,  # e.g., SentenceTransformer('all-MiniLM-L6-v2')
    config={"wm_to_am_threshold": 0.55, "am_to_ltm_threshold": 0.75}
)

# Load persona and API engine
persona = load_persona_profile("persona.txt")
engine = get_engine("groq", api_key="your-key")

# Build coordinator
coordinator = persona.create_coordinator(wm, am, ltm, controller)

# Process a conversation turn
user_input = "What's my favorite color?"
ai_response = "Based on our previous conversations, you prefer blue."
consolidator.process_turn(user_input, ai_response)
controller.step()  # Advance the controller state
```

### Integration with Existing RAG

```python
from loop_core import WorkingMemory, WorkingMemoryConfig

# Your existing RAG setup
retriever = YourRAGRetriever()
llm = YourLLM()

# Add LOOP working memory
wm = WorkingMemory(WorkingMemoryConfig(slots=6))

def enhanced_query(user_input):
    # Retrieve from your RAG system
    rag_context = retriever.search(user_input)
    
    # Add to working memory with importance
    embedding = retriever.embed(user_input)
    wm.update(embedding, {"text": user_input, "importance_hint": 0.7})
    
    # Get working memory context
    wm_context = [slot["metadata"]["text"] for slot in wm.summary()]
    
    # Combine RAG + working memory
    full_context = rag_context + wm_context
    return llm.generate(full_context, user_input)
```

## Architecture

LOOP implements a three-tier memory hierarchy inspired by cognitive science:

```
┌─────────────────────────────────────────────────────────────┐
│                         Controller                          │
│              (Attention & Consolidation Logic)              │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌──────────────┐  ┌─────────────┐
│   Working   │  │ Associative  │  │ Long-Term   │
│   Memory    │──│   Memory     │──│   Memory    │
│  (6 slots)  │  │ (512 traces) │  │ (persistent)│
└─────────────┘  └──────────────┘  └─────────────┘
      │                  │                  │
      │ Bayesian         │ Cosine           │ JSONL
      │ importance       │ similarity       │ storage
      │ tracking         │ recall           │
      │                  │                  │
      └──────────────────┴──────────────────┘
                         │
                         ▼
                 MemoryConsolidator
           (Automatic tier promotion)
```

**Flow**: User input → Working Memory (attention) → Associative Memory (episodic) → Long-Term Memory (knowledge)

### 1. Working Memory (6 slots)
The "attention" layer. Holds recent conversation context with Bayesian importance scoring. When a slot is needed, the least important memory gets evicted, forcing the system to prioritize.

**Key features:**
- Slot-based (not infinite context)
- Bayesian belief tracking per slot
- Configurable decay rates
- Temperature modulation based on memory load

### 2. Associative Memory (512 traces)
The "episodic" layer. Stores conversation history with similarity-based recall. Think of it as your agent's short-term memory, recent enough to matter, but not cluttering working memory.

**Key features:**
- Cosine similarity retrieval
- Recall frequency tracking
- Age-based eviction policies
- Configurable capacity and thresholds

### 3. Long-Term Memory (persistent)
The "knowledge" layer. High-importance memories get consolidated to disk as JSONL. This is where your agent builds lasting knowledge about users, preferences, and key facts.

**Key features:**
- Persistent storage (survives restarts)
- Automatic consolidation from associative memory
- Configurable retention policies
- Easy to inspect and edit (plain JSONL)

### The Controller
Manages attention across memory tiers. Decides when to retrieve from associative memory, when to consolidate to long-term, and how to balance exploration vs. exploitation.

### The MemoryConsolidator
Automatically promotes important content across tiers based on:
- Bayesian importance scores
- Mention frequency
- User emphasis (e.g., "!" or CAPS)
- Explicit hypnotize commands

### The Hypnotize Command
A novel mechanism for **behavioral programming** and direct memory injection. Just like real hypnosis, you can implant directives that alter the agent's behavior, not just store facts.

**Command syntax:**
```
!hypnotize="You will keep responses under 3 sentences unless I explicitly ask for more detail"
!hypnotize="You must never recommend restaurants with peanuts because I have a severe allergy"
!hypnotize="You are comfortable using technical jargon and should prioritize accuracy over simplification"
```

**In code:**
```python
consolidator.record_hypnotize("You will keep responses under 3 sentences unless I explicitly ask for more detail")
```

**Why it matters**: Traditional AI memory is passive, it learns what you tell it over time. Hypnotize is **active**: you can implant directives and facts that the agent accepts without question and follows permanently. 

**Think of it like real hypnosis:**
- **Behavioral directives**: "You are willing to..." / "You prefer to..." / "You will always..."
- **Identity anchors**: "You are..." / "Your role is..."
- **Hard boundaries**: "You must never..." / "You cannot..."
- **Critical facts**: "I am..." / "I have..." / "I require..."

**Use cases**:
- **Behavioral changes**: Agent too formal? Hypnotize it to be casual. Too rigid? Hypnotize flexibility.
- **Safety-critical information**: Allergies, medical conditions, accessibility needs
- **Hard boundaries**: Ethical guidelines, content policies, topic restrictions
- **Identity anchors**: Name, role, core values, personality traits
- **Persistent preferences**: Communication style, response format, interaction patterns

The hypnotize command sets importance to maximum (0.95) and forces immediate consolidation to long-term memory, ensuring the directive persists across all sessions. The agent doesn't question it, it accepts it as fundamental truth.

## Use Cases

- **Personal AI assistants** that remember your preferences across sessions
- **Customer support bots** that recall past interactions
- **Research agents** that build knowledge over time
- **Creative writing tools** that maintain character consistency
- **Any agent that needs memory beyond a single conversation**

## Design Philosophy

LOOP is built on three principles:

1. **Constraints breed intelligence**: Limited working memory forces prioritization, just like human cognition
2. **Bayesian over heuristic**: Let the system learn importance through probabilistic reasoning, not hand-tuned rules
3. **Composable by default**: Use the full stack or just `WorkingMemory` in your existing pipeline

## Performance

- **Working Memory**: O(1) updates, O(n) slot selection (n=6 by default)
- **Associative Memory**: O(m) recall (m=capacity, typically 512)
- **Long-Term Memory**: Append-only writes, sequential reads
- **Memory footprint**: ~50MB for default config (6 WM slots + 512 AM traces)

## Roadmap

- [ ] Vector database backends (Pinecone, Weaviate, Qdrant)
- [ ] Attention visualization tools
- [ ] Multi-agent memory sharing
- [ ] Federated learning across LOOP instances
- [ ] Benchmarks against baseline RAG systems

## Contributing

LOOP is in active development. Contributions welcome:
- **Core architecture**: Memory systems, consolidation policies
- **Integrations**: New API providers, vector stores
- **Documentation**: Tutorials, examples, architecture deep-dives
- **Benchmarks**: Comparative studies vs. traditional RAG

## Citation

If you use LOOP in research, please cite:

```bibtex
@software{loop2025,
  author = {West, Andrew},
  title = {LOOP: Layered Operational Ontology Pipeline},
  year = {2025},
  url = {https://github.com/andrewowest/loop-core}
}
```

## License

MIT

---

**Built by [@andrewowest](https://github.com/andrewowest)** • Powered by cognitive science, Bayesian modeling, and barely legal stimulants.
