# Grok Code

AI coding assistant powered by xAI's Grok, built to match Claude Code's functionality.

## Features

- **Full tool suite**: Bash, Read, Write, Edit, Glob, Grep, Web Search, Web Fetch
- **Vision support**: Read and analyze images (PNG, JPG, GIF, etc.) with automatic model switching
- **Streaming responses**: Real-time output as the model generates
- **Session management**: Save, resume, and continue conversations with smart filtering
- **Auto model selection**: Intelligently switches between fast/coding/reasoning/vision models
- **Cost tracking**: Monitor token usage and API costs with prompt caching
- **Project configuration**: GROK.md files for project-specific and global instructions
- **Task management**: Built-in todo tracking with plan mode
- **Interactive REPL**: Full-featured command-line interface
- **Loop detection**: Prevents infinite tool call loops with automatic recovery

## Installation

Already installed! The `grok` command is available in your PATH.

## Usage

### Command Line Mode

```bash
# Single prompt
grok "Create a hello world program in Python"

# Analyze images
grok "Read screenshot.png and explain the UI issue"

# Session management
grok --resume                          # List sessions for current directory
grok --resume <session-id> "continue"  # Resume specific session
grok -c "continue last conversation"   # Continue most recent session

# Model selection
grok --model grok-2-vision-1212 "Analyze this image"
grok --verbose "Debug with detailed output"

# List all sessions
grok --sessions
```

### Interactive Mode

```bash
grok
```

#### Interactive Commands

- `/clear` - Start new session
- `/save` - Save current session
- `/resume <id>` - Resume specific session
- `/rename <name>` - Name current session
- `/sessions` - List sessions for current directory
- `/sessions all` - List all sessions
- `/cost` - Show token usage and costs
- `/verbose` - Toggle verbose mode
- `/auto` - Toggle auto model selection
- `/model [name]` - View/change model
- `/undo` - Remove last exchange
- `/retry [model]` - Retry last message (optionally with different model)
- `/history` - View conversation history
- `/search <query>` - Web search
- `/plan <task>` - Create implementation plan
- `/exit` or `/quit` - Exit Grok Code

### Options

- `--model MODEL` - Manually specify model (overrides auto-selection)
  - `grok-3-mini` - Fast, cheap ($0.20/1M tokens) - auto-selected for simple queries
  - `grok-code-fast-1` - Coding (default, $5/$15 per 1M) - auto-selected for coding
  - `grok-3` - Balanced general use ($5/$15 per 1M)
  - `grok-4-fast-reasoning` - Advanced reasoning ($5/$15 per 1M) - auto-selected for architecture
  - `grok-2-vision-1212` - Vision (disabled currently, $2/$10 per 1M)
- `--resume [ID]` - Resume session (lists current dir sessions if no ID)
- `-c, --continue` - Continue most recent session
- `--sessions` - List all recent sessions
- `--no-stream` - Disable streaming responses
- `--verbose` - Show detailed tool execution
- `--no-auto-select` - Disable automatic model selection
- `--name NAME` - Name the session
- `-p, --print` - Print mode (output only, no UI)
- `--config-dir DIR` - Custom configuration directory
- `--version` - Show version number

## Tools

Grok Code has access to these tools:

### bash
Execute shell commands

```
grok "List all Python files in the current directory"
```

### read_file
Read file contents with line numbers. **Supports images** - automatically detects and analyzes PNG, JPG, GIF, and other image formats with vision model.

```
grok "Read the file main.py"
grok "Read screenshot.png and explain what's wrong with the UI"
```

### write_file
Create or overwrite files

```
grok "Create a new file called test.txt with 'Hello World'"
```

### edit_file
Edit files by replacing text

```
grok "In config.json, change port from 3000 to 8080"
```

### glob
Find files matching patterns

```
grok "Find all TypeScript files in the src directory"
```

### grep
Search for patterns in files

```
grok "Search for 'TODO' comments in the codebase"
```

## Model Selection

Grok Code supports **both automatic and manual** model selection.

### Automatic Model Selection (Default)

Enabled by default. Analyzes your prompt and automatically chooses the optimal model:

**grok-3-mini** (Fast tier - $0.20/1M):
- Simple queries: "list files", "find files", "count"
- File searches with < 15 words
- Quick operations

**grok-code-fast-1** (Coding tier - $5/$15 per 1M):
- **Default for most tasks**
- All coding operations
- File edits, debugging, refactoring
- Standard development work

**grok-4-fast-reasoning** (Reasoning tier - $5/$15 per 1M):
- Architecture design
- Complex refactoring
- System design, trade-off analysis
- Security audits, migration plans

**Examples:**
```bash
grok "list files"                    # Auto: grok-3-mini
grok "fix this bug"                  # Auto: grok-code-fast-1
grok "design a scalable architecture" # Auto: grok-4-fast-reasoning
```

### Manual Model Selection

Override auto-selection for specific needs:

**Command line:**
```bash
grok --model grok-3-mini "any task"
grok --model grok-3 "balanced work"
```

**Interactive mode:**
```bash
grok
❯ /model grok-3-mini        # Switch to fast model
❯ /model                    # View all options
❯ /auto                     # Toggle auto-selection on/off
```

**Disable auto-selection:**
```bash
grok --no-auto-select --model grok-3 "use grok-3 for everything"
```

In interactive mode: `/auto` toggles between automatic and manual mode.

### When to Use Manual Selection

- **Testing**: Compare model outputs
- **Cost control**: Use grok-3-mini for everything to save money
- **Specific needs**: Force reasoning model for all tasks
- **Consistency**: Keep same model across conversation

## Configuration

### Directory Structure

```
~/.grok/
├── GROK.md               # Global instructions (like CLAUDE.md)
├── sessions/             # Saved conversation sessions
└── history.jsonl         # Global conversation log

.grok/                    # Project-specific (in any directory)
└── GROK.md              # Project-specific instructions
```

### GROK.md Instructions

Create custom instructions for Grok Code:

**Global**: `~/.grok/GROK.md` - applies to all sessions
**Project**: `.grok/GROK.md` - applies only to current project

Example:
```markdown
# Project Guidelines
- Use TypeScript for all new files
- Follow ESLint configuration
- Write tests for all features
- Use functional components in React
```

## Environment Variables

- `GROK_API_KEY` - Your xAI API key (required)

## Examples

```bash
# Create a web server
grok "Create a simple HTTP server in Python that serves the current directory"

# Analyze images
grok "Read screenshot.png and tell me what UI elements are broken"
grok "Compare design.png with the current UI and list differences"

# Session management
grok --resume                    # List sessions for this project
grok -c "Add error handling"     # Continue last session

# Debug code
grok "Read error.log and fix the issues in server.js"

# Refactor code
grok "Refactor the auth.js file to use async/await instead of promises"

# Search codebase
grok "Find all instances where we connect to the database"

# Run tests with cost tracking
grok --verbose "Run the test suite and fix any failures"
# Then check costs:
grok --resume <session-id>
# In interactive mode: /cost

# Web research
grok "Search for best practices for React hooks and implement them in App.js"

# Planning complex tasks
grok "Create a plan for adding user authentication with JWT"
```

## Comparison to Claude Code

Grok Code implements the same core architecture as Claude Code:

| Feature | Claude Code | Grok Code |
|---------|-------------|-----------|
| Streaming responses | ✓ | ✓ |
| Tool calling | ✓ | ✓ |
| File operations | ✓ | ✓ |
| Image analysis | ✓ | ✓ |
| Shell execution | ✓ | ✓ |
| Session management | ✓ | ✓ |
| Auto model selection | ✓ | ✓ |
| Cost tracking | ✓ | ✓ |
| Interactive mode | ✓ | ✓ |
| Project config (GROK.md/CLAUDE.md) | ✓ | ✓ |
| Web search & fetch | ✓ | ✓ |
| Task planning | ✓ | ✓ |
| Loop detection | ✓ | ✓ |

## Version

Grok Code 3.0.0

Powered by xAI (Grok models: mini, code-fast, vision, reasoning)
