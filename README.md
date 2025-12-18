# Grok Code

AI coding assistant powered by xAI's Grok, built to match Claude Code's functionality.

## Features

- **Full tool suite**: Bash, Read, Write, Edit, Glob, Grep
- **Streaming responses**: Real-time output as the model generates
- **Conversation management**: Save, load, and resume conversations
- **Persistent history**: All interactions saved to `~/.grok/history.jsonl`
- **Permissions system**: Control what tools can be executed
- **Interactive REPL**: Full-featured command-line interface
- **Settings management**: Customizable configuration

## Installation

Already installed! The `grok` command is available in your PATH.

## Usage

### Command Line Mode

```bash
# Single prompt
grok "Create a hello world program in Python"

# With options
grok --model grok-3 "Explain this code"
grok --no-stream "Quick question"

# Load previous conversation
grok --context conversation.json "Continue where we left off"
```

### Interactive Mode

```bash
grok
```

#### Interactive Commands

- `/clear` - Clear conversation history
- `/save <filename>` - Save conversation to JSON file
- `/load <filename>` - Load conversation from JSON file
- `/exit` or `/quit` - Exit Grok Code

### Options

- `--model MODEL` - Specify model (default: grok-3)
- `--no-stream` - Disable streaming responses
- `--context FILE` - Load conversation context from file
- `--config-dir DIR` - Use custom configuration directory
- `--version` - Show version number

## Tools

Grok Code has access to these tools:

### bash
Execute shell commands

```
grok "List all Python files in the current directory"
```

### read_file
Read file contents with line numbers

```
grok "Read the file main.py"
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

## Configuration

### Directory Structure

```
~/.grok/
├── settings.json           # Global settings
├── settings.local.json     # Local overrides
├── history.jsonl          # Conversation history
├── debug/                 # Debug logs
├── file-history/          # File change history
├── plans/                 # Saved plans
├── todos/                 # Todo lists
├── shell-snapshots/       # Shell state snapshots
└── downloads/             # Downloaded files
```

### Settings

Edit `~/.grok/settings.local.json`:

```json
{
  "permissions": {
    "allow": ["Bash:*", "Read:*", "Write:*"],
    "deny": [],
    "defaultMode": "prompt"
  }
}
```

#### Permission Modes

- `prompt` - Ask before executing (default)
- `allow` - Auto-allow
- `deny` - Auto-deny

## Environment Variables

- `GROK_API_KEY` - Your xAI API key (required)

## Examples

```bash
# Create a web server
grok "Create a simple HTTP server in Python that serves the current directory"

# Debug code
grok "Why is this function returning undefined?" --context debug-session.json

# Refactor code
grok "Refactor the auth.js file to use async/await instead of promises"

# Search codebase
grok "Find all instances where we connect to the database"

# Run tests
grok "Run the test suite and fix any failures"
```

## Comparison to Claude Code

Grok Code implements the same core architecture as Claude Code:

| Feature | Claude Code | Grok Code |
|---------|-------------|-----------|
| Streaming responses | ✓ | ✓ |
| Tool calling | ✓ | ✓ |
| File operations | ✓ | ✓ |
| Shell execution | ✓ | ✓ |
| Conversation history | ✓ | ✓ |
| Permissions system | ✓ | ✓ |
| Interactive mode | ✓ | ✓ |
| Settings management | ✓ | ✓ |

## Version

Grok Code 1.0.0

Powered by xAI Grok-3
