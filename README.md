# Agentic Cognitive Engine (ACE)

A sophisticated agentic AI system built with LangGraph that iteratively learns and improves through task execution, reflection, and curation. The system maintains a dynamic playbook of insights and uses vector-based memory for context-aware decision making.

## Key Features

- **Iterative Learning**: Multi-node agent architecture (Generator → Reflector → Curator → Consolidator → Human Feedback)
- **Vector Memory**: FAISS-powered vector database for storing playbook bullets and task states
- **Thread-Safe Operations**: Cross-platform file locking for concurrent FAISS index access
- **Checkpointing**: LangGraph checkpointing for resumable task execution
- **Dynamic Playbook**: Self-updating knowledge base with helpful/harmful feedback tracking

## Architecture

### Core Components

#### `implementing_ace5.py`
The main agent implementation featuring:
- **Generator Node**: Creates initial task responses using playbook and memory context
- **Reflector Node**: Analyzes generator output for errors and insights
- **Curator Node**: Decides whether to add/update playbook entries
- **Consolidator Node**: Manages routing and playbook updates
- **Human Feedback Node**: Interactive approval and feedback collection

#### `SharedFAISSManager.py`
Thread/process-safe FAISS index manager with:
- Cross-platform file locking (portalocker on Windows, fcntl on Unix)
- Automatic index persistence and loading
- Concurrent access protection

### Data Flow
1. User submits task → Generator creates initial response
2. Reflector analyzes for improvements and tags playbook bullets
3. Curator scores output and suggests playbook updates
4. Consolidator routes based on score (retry or human review)
5. Human provides feedback → System learns and updates memory

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agentic_sol
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myagenticvenv
   # On Windows:
   myagenticvenv\Scripts\activate
   # On Unix/Mac:
   source myagenticvenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   Create a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   LLM_MODEL=gpt-4o-mini  # or your preferred model
   ```

## Usage

### Basic Example

```python
from implementing_ace5 import MultiCollectionVectorDB, build_ace_graph, LangGraphState
from langgraph.checkpoint.memory import MemorySaver

# Initialize vector database
vdb = MultiCollectionVectorDB()

# Build the agent graph
graph = build_ace_graph(vdb)
checkpointer = MemorySaver()

# Create initial state
initial_state = LangGraphState(
    generator_inputs={"user_task": "Explain quantum computing basics"}
)

# Run the agent
config = {"configurable": {"thread_id": "example_task"}}
result = graph.invoke(initial_state, config=config, checkpointer=checkpointer)

print("Final Answer:", result.generator_output["final_ans"])
```

### Advanced Usage

The system supports:
- **Resume interrupted tasks** via LangGraph checkpoints
- **Multi-collection vector storage** (playbook, states)
- **Interactive prompt customization** for each node
- **Similarity-based playbook deduplication**

## Dependencies

Key packages include:
- `langgraph` - Graph-based agent orchestration
- `langchain-openai` - OpenAI LLM integration
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Text embedding generation
- `pydantic` - Data validation and serialization
- `python-dotenv` - Environment variable management

See `requirements.txt` for complete list.

## Project Structure

```
agentic_sol/
├── implementing_ace5.py          # Main agent implementation
├── SharedFAISSManager.py         # Thread-safe FAISS manager
├── requirements.txt              # Python dependencies
├── vectors_playbook.npy          # Playbook vector embeddings
├── vectors_states.npy            # States vector embeddings
├── metadata_playbook.pkl         # Playbook metadata
├── metadata_states.pkl           # States metadata
├── playbook_bullet.json          # Bullet ID tracker
└── myagenticvenv/                # Virtual environment
```

## Configuration

### Vector Collections
- **playbook**: Stores reusable insights and best practices
- **states**: Maintains task execution history and learning data

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `LLM_MODEL`: Model to use (default: gpt-4o-mini)

## Contributing

1. Follow the existing code structure and naming conventions
2. Add type hints for new functions
3. Update docstrings for complex logic
4. Test with both simple and complex task scenarios
