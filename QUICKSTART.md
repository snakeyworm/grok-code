# Grok Code Quick Start

## Installation Complete

Grok Code is installed and ready to use! Command: `grok`

## Basic Usage

### One-off Commands
```bash
grok "create a hello world script"
grok "find all Python files in this directory"
grok "explain what this code does" --context conversation.json
```

### Interactive Mode
```bash
grok
```

Then type commands naturally:
```
> create a web server in Python
> read main.py and add error handling
> run the tests
> /exit
```

## Interactive Commands

- `/clear` - Clear conversation history
- `/save <file>` - Save conversation
- `/load <file>` - Load previous conversation
- `/verbose` - Toggle verbose tool output
- `/exit` - Quit

## Key Features

**Smart File Operations**
- Automatically uses `/tmp` for test files
- Prefers specialized tools over bash for file operations
- Always reads files before editing them

**Tool Chaining**
- Grok chains multiple tools to complete complex tasks
- Handles errors gracefully and adapts approach
- Continues until task completion

**Context Aware**
- Knows your working directory
- Maintains conversation history
- Can resume previous sessions

## Options

```bash
--model grok-3         # Use specific model
--no-stream           # Disable streaming
--context file.json   # Load conversation
--verbose             # Show tool execution
--config-dir ~/.grok  # Custom config directory
```

## Examples

**Create and test a script:**
```bash
grok "create a Python script that sorts a list of numbers, then test it"
```

**Find and modify code:**
```bash
grok "find all TODO comments and create a summary file"
```

**Debug and fix:**
```bash
grok "read app.py and fix any syntax errors"
```

**Complex workflows:**
```bash
grok "analyze all JavaScript files, identify potential bugs, and create a report"
```

## Configuration

Config directory: `~/.grok/`
- `settings.json` - Global settings
- `settings.local.json` - Local overrides
- `history.jsonl` - Full conversation history

## Tips

1. **Be specific but concise** - "create a test file in /tmp" works better than "write me a file"
2. **Let Grok use tools** - It will chain bash, read_file, edit_file, etc. automatically
3. **Use /verbose** - See exactly what tools are being called
4. **Save important sessions** - Use `/save` before experimenting
5. **Resume context** - Use `--context file.json` to continue previous work

## Troubleshooting

**API Error:**
```bash
export GROK_API_KEY='your-key-here'
```

**Tool permission issues:**
Edit `~/.grok/settings.local.json` permissions

**Unexpected behavior:**
Try with `--verbose` to see tool execution

## Tools Available

- `bash` - Shell commands (git, npm, etc.)
- `read_file` - Read files with line numbers
- `write_file` - Create/overwrite files
- `edit_file` - Find and replace text
- `glob` - Pattern-based file finding
- `grep` - Search file contents

Grok automatically chooses the right tools for each task.
