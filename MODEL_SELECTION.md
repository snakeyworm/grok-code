# Grok Code - Dynamic Model Selection

## Overview

Grok Code 2.0 implements Claude Code's intelligent model selection strategy, automatically choosing the optimal Grok model for each task based on complexity.

## Model Tier Mapping

### Claude Code → Grok Code

| Claude Tier | Grok Model | Use Case | Speed | Cost |
|-------------|------------|----------|-------|------|
| **Haiku** | grok-3-mini | Fast, simple tasks | Fastest | $0.20/1M |
| **Sonnet** | grok-code-fast-1 | Standard coding (DEFAULT) | 92 tok/sec | $0.20/1M in, $1.50/1M out |
| **Opus** | grok-4-fast-reasoning | Complex reasoning | Fast | $0.20/1M in, $0.50/1M out |

## Model Selection Details

### grok-3-mini (Fast Tier)
**Characteristics:**
- Fastest responses
- Lowest cost
- Optimized for simple queries

**Auto-selected for:**
- File listings: `list files`, `find files`
- Simple searches: `show me`, `what files`
- Count operations: `how many`
- Short queries (< 15 words)

**Examples:**
```bash
grok "list all Python files"          # → grok-3-mini
grok "find files matching *.js"       # → grok-3-mini
grok "how many files in this dir?"    # → grok-3-mini
```

### grok-code-fast-1 (Coding Tier - DEFAULT)
**Characteristics:**
- 314B MoE architecture
- 92 tokens/sec - fastest coding model
- Specialized for agentic coding
- Scored 70.8% on SWE-Bench-Verified
- 256K context, 10K output

**Auto-selected for:**
- All coding tasks (default for coding assistant)
- File operations (read, write, edit)
- Script creation
- Code refactoring
- Test writing
- Debugging

**Examples:**
```bash
grok "create a web server"            # → grok-code-fast-1
grok "write me a random file"         # → grok-code-fast-1
grok "fix the bug in main.py"         # → grok-code-fast-1
grok "add error handling"             # → grok-code-fast-1
```

### grok-4-fast-reasoning (Reasoning Tier)
**Characteristics:**
- Advanced reasoning capabilities
- 40% fewer thinking tokens than Grok 4
- 98% lower cost for same performance
- Outperforms competitors on agentic benchmarks

**Auto-selected for:**
- Architectural design: `design system`, `architecture`
- Complex refactoring: `refactor entire`, `redesign`
- Performance optimization: `scalability`, `optimization`
- Security audits: `security audit`
- Algorithm design: `complex logic`, `algorithm design`
- Migration planning: `migration plan`, `trade-offs`

**Examples:**
```bash
grok "design a scalable microservices architecture"  # → grok-4-fast-reasoning
grok "best approach for caching strategy?"           # → grok-4-fast-reasoning
grok "refactor entire authentication system"         # → grok-4-fast-reasoning
grok "performance optimization for database"         # → grok-4-fast-reasoning
```

## Multi-Agent Architecture

### spawn_task Tool

Like Claude Code's Task tool, Grok Code can spawn sub-agents with different models:

```python
# Automatically spawned by main agent
spawn_task(
    prompt="Search for all TODO comments",
    model="grok-3-mini",           # Fast search
    description="Finding TODOs"
)

spawn_task(
    prompt="Implement user authentication",
    model="grok-code-fast-1",      # Standard coding
    description="Auth implementation"
)

spawn_task(
    prompt="Design database schema for high scalability",
    model="grok-4-fast-reasoning",  # Complex design
    description="Schema design"
)
```

### How It Works

1. **Main agent** runs on grok-code-fast-1 (default)
2. **Detects complex sub-tasks** that need different models
3. **Spawns sub-agent** with appropriate model tier
4. **Sub-agent executes** and returns result
5. **Main agent integrates** result into response

### Benefits

- **Cost optimization**: Use cheap models for simple tasks
- **Speed optimization**: Fast models for quick operations
- **Quality optimization**: Advanced reasoning for complex decisions
- **Parallel execution**: Multiple sub-agents can run different tasks

## Usage

### Automatic Mode (Default)

```bash
grok "your task"  # Auto-selects best model
```

Grok Code automatically analyzes your prompt and selects:
- Fast tier for simple operations
- Coding tier for standard development
- Reasoning tier for complex architecture

### Manual Override

```bash
# Force specific model
grok --model grok-4-fast-reasoning "create hello world"

# Disable auto-selection
grok --no-auto-select "your task"
```

### Interactive Mode

```bash
grok  # Enter interactive mode

> /model                    # Show current model
> /model grok-3-mini       # Change model
> /auto                     # Toggle auto-selection
> /verbose                  # See model selection decisions
```

## Verbose Mode

Enable verbose to see model selection in action:

```bash
grok --verbose "design microservices architecture"
```

Output:
```
[Auto-selected model: grok-4-fast-reasoning]
[Spawning sub-agent: grok-4-fast-reasoning for 'Architectural design']
...
```

## Configuration

### Enable/Disable Auto-Selection

```bash
# Command line
grok --no-auto-select "task"  # Disable for this task

# Interactive mode
> /auto  # Toggle on/off
```

### Set Default Model

```bash
# Command line
grok --model grok-3 "task"

# Interactive mode
> /model grok-3
```

## Performance Comparison

| Task Type | Old (single model) | New (multi-model) | Improvement |
|-----------|-------------------|-------------------|-------------|
| Simple search | grok-3 | grok-3-mini | 2x faster |
| Standard coding | grok-3 | grok-code-fast-1 | 3x faster, better quality |
| Architecture design | grok-3 | grok-4-fast-reasoning | Higher quality |

## Cost Analysis

**Example workflow:** "Design auth system, implement it, test it"

**Old (single model):**
- All on grok-3: $3/1M in, $15/1M out

**New (multi-model):**
- Design: grok-4-fast-reasoning ($0.20 in, $0.50 out)
- Implement: grok-code-fast-1 ($0.20 in, $1.50 out)
- Test search: grok-3-mini ($0.20 in, $0.20 out)
- **Total: ~90% cost reduction with better quality**

## Technical Details

### Model Selection Algorithm

Located in `ModelSelector` class:

```python
def select_model_for_task(prompt: str, current_model: str) -> str:
    # 1. Check for simple operations → grok-3-mini
    # 2. Check for complex reasoning → grok-4-fast-reasoning
    # 3. Default to coding model → grok-code-fast-1
```

### Integration with Tools

Sub-agents have access to all tools:
- bash
- read_file
- write_file
- edit_file
- glob
- grep

They **cannot** spawn further sub-agents (prevents infinite recursion).

## Best Practices

1. **Trust auto-selection** - It's optimized for your use case
2. **Use verbose mode** - When debugging or learning
3. **Manual override rarely** - Only when you know better
4. **Monitor costs** - Check ~/.grok/history.jsonl for model usage

## Troubleshooting

**Q: Why did it select the wrong model?**
A: Enable `--verbose` to see selection reasoning. You can override with `--model`.

**Q: Can I customize selection logic?**
A: Edit `ModelSelector.select_model_for_task()` in the source code.

**Q: How do I force a specific model for all tasks?**
A: Use `--model` flag and `--no-auto-select`.

## Future Enhancements

Planned features:
- Learning from user feedback
- Per-project model preferences
- Cost tracking and budgets
- Model performance analytics

---

**Version:** Grok Code 2.0.0
**Last Updated:** 2025-12-16
