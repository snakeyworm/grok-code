#!/usr/bin/env python3
import os
import sys
import json
import re
import subprocess
import glob as glob_module
import readline
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Any
from datetime import datetime
import tempfile
import platform
import threading
import time

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests


SYSTEM_PROMPT = """You are Grok Code, an AI coding assistant with access to powerful tools for software development.

You have access to these tools:
- bash: Execute shell commands (git, npm, docker, etc. - NOT for file operations)
- read_file: Read file contents with line numbers
- write_file: Create or overwrite files
- edit_file: Replace text in files (exact string matching)
- glob: Find files by pattern (e.g., **/*.js)
- grep: Search file contents (regex patterns)
- spawn_task: Spawn a sub-agent with different model for specialized tasks

Key principles:
- Use specialized tools (read_file, write_file, edit_file) for file operations, NOT bash
- Always read files before editing or modifying them
- For file edits, use exact string matching - match indentation precisely
- Be concise and direct in responses
- Execute tasks fully - no partial solutions or TODOs
- When creating files, use appropriate paths (prefer /tmp for test files)
- Chain tools together to complete multi-step tasks
- Focus on the task at hand, avoid over-engineering
- Use spawn_task for complex sub-tasks that would benefit from specialized models

Response style:
- Brief, technical communication
- Show tool usage directly without excessive explanation
- Let the tool results speak for themselves
- Only provide context when necessary

Current model: {current_model}
Working directory: {cwd}
Platform: {platform}
Date: {date}
"""


class ModelSelector:
    """Intelligent model selection based on task complexity"""

    # Model tier mapping (Claude Code equivalent)
    FAST_MODEL = "grok-3-mini"           # Haiku tier: simple, fast, cheap
    CODING_MODEL = "grok-code-fast-1"    # Sonnet tier: balanced coding (DEFAULT)
    REASONING_MODEL = "grok-4-fast-reasoning"  # Opus tier: complex reasoning

    @staticmethod
    def select_model_for_task(prompt: str, current_model: str = None) -> str:
        """
        Select optimal model based on task complexity

        Fast tier (grok-3-mini):
        - File searches, listings
        - Simple queries
        - Quick reads

        Coding tier (grok-code-fast-1):
        - All coding tasks (DEFAULT for coding assistant)
        - File operations
        - Standard development work

        Reasoning tier (grok-4-fast-reasoning):
        - Architecture design
        - Complex refactoring
        - Critical decisions
        """

        prompt_lower = prompt.lower()

        # Fast tier: Simple search/list operations
        fast_indicators = [
            'list files', 'find files', 'search for',
            'show me', 'what files', 'glob', 'ls',
            'count', 'how many'
        ]
        if any(indicator in prompt_lower for indicator in fast_indicators):
            if len(prompt.split()) < 15:  # Short, simple queries
                return ModelSelector.FAST_MODEL

        # Reasoning tier: Complex architectural tasks
        reasoning_indicators = [
            'architecture', 'design system', 'refactor entire',
            'redesign', 'best approach', 'trade-offs',
            'scalability', 'performance optimization',
            'security audit', 'migration plan',
            'complex logic', 'algorithm design'
        ]
        if any(indicator in prompt_lower for indicator in reasoning_indicators):
            return ModelSelector.REASONING_MODEL

        # Default to coding model for all other tasks
        # This is a CODING ASSISTANT, so grok-code-fast-1 is the workhorse
        return current_model or ModelSelector.CODING_MODEL


class Tool:
    def __init__(self, name: str):
        self.name = name

    def execute(self, **kwargs) -> str:
        raise NotImplementedError


class BashTool(Tool):
    def __init__(self):
        super().__init__("Bash")

    def execute(self, command: str, timeout: int = 120) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            output = []
            if result.stdout:
                output.append(result.stdout.rstrip())
            if result.stderr:
                output.append(result.stderr.rstrip())
            if result.returncode != 0:
                output.append(f"Exit code: {result.returncode}")
            return "\n".join(output) if output else "Command completed successfully"
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error: {str(e)}"


class ReadTool(Tool):
    def __init__(self):
        super().__init__("Read")

    def execute(self, file_path: str, offset: int = 0, limit: int = 0) -> str:
        try:
            file_path = os.path.expanduser(file_path)
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            total_lines = len(lines)

            if offset > 0:
                lines = lines[offset:]
            if limit > 0:
                lines = lines[:limit]

            numbered_lines = [f"{i+1+offset:5}→{line.rstrip()}" for i, line in enumerate(lines)]
            result = "\n".join(numbered_lines)

            if offset > 0 or (limit > 0 and total_lines > offset + limit):
                result = f"File: {file_path} (showing lines {offset+1}-{offset+len(lines)} of {total_lines})\n{result}"

            return result
        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except PermissionError:
            return f"Error: Permission denied: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class WriteTool(Tool):
    def __init__(self):
        super().__init__("Write")

    def execute(self, file_path: str, content: str) -> str:
        try:
            file_path = os.path.expanduser(file_path)
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            lines = content.count('\n') + 1
            return f"Wrote {lines} lines to {file_path}"
        except PermissionError:
            return f"Error: Permission denied: {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


class EditTool(Tool):
    def __init__(self):
        super().__init__("Edit")

    def execute(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        try:
            file_path = os.path.expanduser(file_path)

            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            count = content.count(old_string)

            if count == 0:
                return f"Error: String not found in {file_path}"

            if count > 1 and not replace_all:
                return f"Error: String appears {count} times. Set replace_all=true to replace all occurrences"

            if replace_all:
                new_content = content.replace(old_string, new_string)
            else:
                new_content = content.replace(old_string, new_string, 1)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            replaced = count if replace_all else 1
            return f"Replaced {replaced} occurrence(s) in {file_path}"
        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except PermissionError:
            return f"Error: Permission denied: {file_path}"
        except Exception as e:
            return f"Error editing file: {str(e)}"


class GlobTool(Tool):
    def __init__(self):
        super().__init__("Glob")

    def execute(self, pattern: str, path: str = ".") -> str:
        try:
            path = os.path.expanduser(path)
            full_pattern = os.path.join(path, pattern)
            matches = sorted(glob_module.glob(full_pattern, recursive=True))

            if not matches:
                return f"No files matching pattern: {pattern}"

            return "\n".join(matches[:500])
        except Exception as e:
            return f"Error: {str(e)}"


class GrepTool(Tool):
    def __init__(self):
        super().__init__("Grep")

    def execute(self, pattern: str, path: str = ".",
                output_mode: str = "files_with_matches",
                glob: str = None, case_insensitive: bool = False,
                context_lines: int = 0) -> str:
        try:
            path = os.path.expanduser(path)

            has_rg = subprocess.run(["which", "rg"], capture_output=True).returncode == 0

            if has_rg:
                cmd = ["rg"]
            else:
                cmd = ["grep", "-r"]

            if case_insensitive:
                cmd.append("-i")

            if output_mode == "files_with_matches":
                cmd.append("-l")
            elif output_mode == "count":
                cmd.append("-c")
            else:
                cmd.append("-n")
                if context_lines > 0:
                    cmd.extend(["-C", str(context_lines)])

            if glob and has_rg:
                cmd.extend(["--glob", glob])

            cmd.append(pattern)
            cmd.append(path)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.stdout:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 500:
                    return "\n".join(lines[:500]) + f"\n... ({len(lines) - 500} more results)"
                return result.stdout.strip()

            return "No matches found"
        except subprocess.TimeoutExpired:
            return "Error: Search timed out"
        except Exception as e:
            return f"Error: {str(e)}"


class SpawnTaskTool(Tool):
    """Spawn sub-agent with different model (like Claude's Task tool)"""

    def __init__(self, parent_client):
        super().__init__("SpawnTask")
        self.parent_client = parent_client

    def execute(self, prompt: str, model: str = None, description: str = "") -> str:
        """
        Spawn a sub-agent to handle a task

        Args:
            prompt: Task for the sub-agent
            model: Model to use (auto-selected if not specified)
            description: Short description of task (for logging)
        """
        try:
            # Auto-select model if not specified
            if not model:
                model = ModelSelector.select_model_for_task(prompt, self.parent_client.model)

            if self.parent_client.verbose:
                desc = description or prompt[:50]
                print(f"\n[Spawning sub-agent: {model} for '{desc}']", file=sys.stderr)

            # Create sub-client with same config but different model
            sub_client = GrokClient(
                self.parent_client.api_key,
                self.parent_client.config_dir,
                model=model
            )
            sub_client.verbose = self.parent_client.verbose

            # Execute task and collect result
            result = []
            for chunk in sub_client.chat(prompt, stream=False):
                result.append(chunk)

            return "".join(result)

        except Exception as e:
            return f"Error spawning task: {str(e)}"


class ToolRegistry:
    def __init__(self, parent_client=None):
        self.parent_client = parent_client
        self.tools = {
            "bash": BashTool(),
            "read_file": ReadTool(),
            "write_file": WriteTool(),
            "edit_file": EditTool(),
            "glob": GlobTool(),
            "grep": GrepTool(),
        }

        if parent_client:
            self.tools["spawn_task"] = SpawnTaskTool(parent_client)

    def get_tool_definitions(self) -> List[Dict]:
        defs = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a bash command. Use for git, npm, docker, tests, etc. DO NOT use for file operations - use read_file, write_file, edit_file instead.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The bash command to execute"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds (default: 120)"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents with line numbers. Always read before editing.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to file (absolute or relative)"},
                            "offset": {"type": "integer", "description": "Line number to start from (0-indexed)"},
                            "limit": {"type": "integer", "description": "Number of lines to read"}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file (creates or overwrites). Use /tmp for test files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to file"},
                            "content": {"type": "string", "description": "Content to write"}
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Edit file by replacing exact text. Match indentation precisely.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to file"},
                            "old_string": {"type": "string", "description": "Exact text to replace"},
                            "new_string": {"type": "string", "description": "New text"},
                            "replace_all": {"type": "boolean", "description": "Replace all occurrences"}
                        },
                        "required": ["file_path", "old_string", "new_string"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "glob",
                    "description": "Find files by pattern (e.g., **/*.js, src/**/*.ts)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Glob pattern"},
                            "path": {"type": "string", "description": "Base directory (default: .)"}
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "grep",
                    "description": "Search files for regex patterns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Regex pattern"},
                            "path": {"type": "string", "description": "Directory to search"},
                            "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]},
                            "glob": {"type": "string", "description": "Filter by glob pattern"},
                            "case_insensitive": {"type": "boolean", "description": "Case insensitive"}
                        },
                        "required": ["pattern"]
                    }
                }
            }
        ]

        # Only add spawn_task if parent_client exists (not for sub-agents)
        if self.parent_client:
            defs.append({
                "type": "function",
                "function": {
                    "name": "spawn_task",
                    "description": "Spawn a sub-agent with different model for specialized tasks. Use 'fast' model for simple searches, 'coding' for standard tasks, 'reasoning' for complex architectural decisions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Task for the sub-agent"},
                            "model": {
                                "type": "string",
                                "enum": ["grok-3-mini", "grok-code-fast-1", "grok-4-fast-reasoning"],
                                "description": "Model tier: grok-3-mini (fast), grok-code-fast-1 (coding), grok-4-fast-reasoning (reasoning)"
                            },
                            "description": {"type": "string", "description": "Short description of task"}
                        },
                        "required": ["prompt"]
                    }
                }
            })

        return defs

    def execute(self, tool_name: str, arguments: Dict) -> str:
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Error: Unknown tool: {tool_name}"
        try:
            return tool.execute(**arguments)
        except TypeError as e:
            return f"Error: Invalid arguments for {tool_name}: {str(e)}"
        except Exception as e:
            return f"Error in {tool_name}: {str(e)}"


class ConversationHistory:
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.config_dir / "history.jsonl"
        self.messages: List[Dict] = []

    def add_message(self, message: Dict):
        self.messages.append(message)
        self._save_to_history(message)

    def _save_to_history(self, message: Dict):
        try:
            with open(self.history_file, 'a') as f:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "message": message,
                    "cwd": os.getcwd()
                }
                f.write(json.dumps(entry) + "\n")
        except:
            pass

    def load_context(self, context_file: str):
        with open(context_file) as f:
            self.messages = json.load(f)

    def save_context(self, context_file: str):
        with open(context_file, 'w') as f:
            json.dump(self.messages, f, indent=2)

    def clear(self):
        self.messages = []


class GrokClient:
    def __init__(self, api_key: str, config_dir: str, model: str = None):
        self.api_key = api_key
        # Default to coding model since this is a coding assistant
        self.model = model or ModelSelector.CODING_MODEL
        self.base_url = "https://api.x.ai/v1"
        self.config_dir = config_dir
        self.history = ConversationHistory(config_dir)
        self.tool_registry = ToolRegistry(parent_client=self)
        self.verbose = False
        self.auto_select_model = True  # Enable dynamic model selection

    def get_system_message(self) -> str:
        return SYSTEM_PROMPT.format(
            current_model=self.model,
            cwd=os.getcwd(),
            platform=f"{platform.system()} {platform.release()}",
            date=datetime.now().strftime("%Y-%m-%d")
        )

    def chat(self, user_message: str, stream: bool = True) -> Iterator[str]:
        # Auto-select model for this specific task if enabled
        if self.auto_select_model:
            selected_model = ModelSelector.select_model_for_task(user_message, self.model)
            if selected_model != self.model and self.verbose:
                print(f"\n[Auto-selected model: {selected_model}]", file=sys.stderr)
            working_model = selected_model
        else:
            working_model = self.model

        self.history.add_message({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.get_system_message()}] + self.history.messages

        max_iterations = 25  # Match Claude Code's limit for complex tasks
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Warn when approaching limit
            if iteration >= max_iterations - 3 and self.verbose:
                print(f"\n[Warning: {max_iterations - iteration} iterations remaining]", file=sys.stderr)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": working_model,
                "messages": messages,
                "tools": self.tool_registry.get_tool_definitions(),
                "stream": stream,
                "temperature": 0.7
            }

            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    stream=stream,
                    timeout=120
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                yield f"\nError calling Grok API: {str(e)}\n"
                return

            if stream:
                needs_continuation = yield from self._handle_stream_response(response, messages)
            else:
                needs_continuation = yield from self._handle_non_stream_response(response, messages)

            if not needs_continuation:
                break

        if iteration >= max_iterations:
            yield f"\n\n[Max iterations ({max_iterations}) reached. Task may be too complex or stuck in a loop. Try breaking it into smaller tasks or use --verbose to debug.]"

    def _handle_stream_response(self, response, messages: List[Dict]) -> bool:
        full_content = ""
        tool_calls_data = {}
        current_message = {"role": "assistant", "content": ""}

        for line in response.iter_lines():
            if not line:
                continue

            line_text = line.decode('utf-8')
            if not line_text.startswith("data: "):
                continue

            data_str = line_text[6:]
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0]["delta"]

                if "content" in delta and delta["content"]:
                    content_piece = delta["content"]
                    full_content += content_piece
                    yield content_piece

                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_data:
                            tool_calls_data[idx] = {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            }

                        if "id" in tc:
                            tool_calls_data[idx]["id"] = tc["id"]
                        if "function" in tc:
                            if "name" in tc["function"]:
                                tool_calls_data[idx]["function"]["name"] = tc["function"]["name"]
                            if "arguments" in tc["function"]:
                                tool_calls_data[idx]["function"]["arguments"] += tc["function"]["arguments"]

            except json.JSONDecodeError:
                continue

        if tool_calls_data:
            tool_calls = [tool_calls_data[i] for i in sorted(tool_calls_data.keys())]
            current_message["tool_calls"] = tool_calls
            current_message["content"] = full_content if full_content else None
            self.history.add_message(current_message)
            messages.append(current_message)

            yield "\n"
            for tool_call in tool_calls:
                yield from self._execute_tool_call(tool_call, messages)

            return True

        if full_content:
            current_message["content"] = full_content
            self.history.add_message(current_message)
            messages.append(current_message)

        return False

    def _handle_non_stream_response(self, response, messages: List[Dict]) -> bool:
        data = response.json()
        message = data["choices"][0]["message"]

        if message.get("tool_calls"):
            self.history.add_message(message)
            messages.append(message)

            for tool_call in message["tool_calls"]:
                yield from self._execute_tool_call(tool_call, messages)

            return True

        if message.get("content"):
            self.history.add_message(message)
            messages.append(message)
            yield message["content"]

        return False

    def _execute_tool_call(self, tool_call: Dict, messages: List[Dict]) -> Iterator[str]:
        tool_name = tool_call["function"]["name"]

        try:
            arguments = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError as e:
            result = f"Error: Invalid tool arguments: {str(e)}"
            self._add_tool_result(tool_call["id"], result, messages)
            yield f"\n[Tool error: {tool_name}]\n"
            return

        if self.verbose:
            yield f"\n[{tool_name}]\n"

        result = self.tool_registry.execute(tool_name, arguments)

        self._add_tool_result(tool_call["id"], result, messages)

        if result.startswith("Error:"):
            yield f"\n{result}\n"

    def _add_tool_result(self, tool_call_id: str, result: str, messages: List[Dict]):
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        }
        self.history.add_message(tool_message)
        messages.append(tool_message)


def interactive_mode(client: GrokClient):
    print("Grok Code - AI Coding Assistant")
    print(f"Model: {client.model} | Working directory: {os.getcwd()}")
    print(f"Auto-select: {'enabled' if client.auto_select_model else 'disabled'}")
    print("\nCommands: /clear /save /load /verbose /auto /model /exit\n")

    while True:
        try:
            user_input = input("\033[1;36m>\033[0m ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['/exit', '/quit']:
                break

            if user_input == '/clear':
                client.history.clear()
                print("✓ Cleared conversation")
                continue

            if user_input.startswith('/save'):
                parts = user_input.split(maxsplit=1)
                filename = parts[1] if len(parts) > 1 else "conversation.json"
                client.history.save_context(filename)
                print(f"✓ Saved to {filename}")
                continue

            if user_input.startswith('/load'):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: /load <filename>")
                    continue
                client.history.load_context(parts[1])
                print(f"✓ Loaded from {parts[1]}")
                continue

            if user_input.startswith('/verbose'):
                client.verbose = not client.verbose
                print(f"✓ Verbose mode: {'on' if client.verbose else 'off'}")
                continue

            if user_input.startswith('/auto'):
                client.auto_select_model = not client.auto_select_model
                print(f"✓ Auto model selection: {'enabled' if client.auto_select_model else 'disabled'}")
                continue

            if user_input.startswith('/model'):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print(f"Current model: {client.model}")
                    print("\nAvailable models:")
                    print("  grok-3-mini           - Fast, cheap")
                    print("  grok-code-fast-1      - Coding (default)")
                    print("  grok-3                - Balanced")
                    print("  grok-4-fast-reasoning - Advanced reasoning")
                    continue
                client.model = parts[1]
                print(f"✓ Model set to: {client.model}")
                continue

            print()
            for chunk in client.chat(user_input, stream=True):
                print(chunk, end='', flush=True)
            print()

        except KeyboardInterrupt:
            print("\n^C")
        except EOFError:
            break
        except Exception as e:
            print(f"\nError: {e}")

    print("\nGoodbye!")


def main():
    import argparse

    config_dir = os.path.expanduser("~/.grok")

    parser = argparse.ArgumentParser(
        description='Grok Code - AI coding assistant',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('prompt', nargs='*', help='Prompt for Grok')
    parser.add_argument('--model', help='Model to use (default: auto-select grok-code-fast-1)')
    parser.add_argument('--no-stream', action='store_true', help='Disable streaming')
    parser.add_argument('--context', help='Load conversation from file')
    parser.add_argument('--version', action='version', version='Grok Code 2.0.0')
    parser.add_argument('--config-dir', default=config_dir, help='Config directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose tool output')
    parser.add_argument('--no-auto-select', action='store_true', help='Disable automatic model selection')

    args = parser.parse_args()

    api_key = os.environ.get('GROK_API_KEY')
    if not api_key:
        print("Error: GROK_API_KEY not set")
        print("Usage: export GROK_API_KEY='your-key'")
        sys.exit(1)

    client = GrokClient(api_key, args.config_dir, model=args.model)
    client.verbose = args.verbose
    client.auto_select_model = not args.no_auto_select

    if args.context and os.path.exists(args.context):
        client.history.load_context(args.context)

    if args.prompt:
        prompt = ' '.join(args.prompt)
        for chunk in client.chat(prompt, stream=not args.no_stream):
            print(chunk, end='', flush=True)
        print()
    else:
        interactive_mode(client)


if __name__ == '__main__':
    main()
