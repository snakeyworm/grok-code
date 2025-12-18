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
import uuid

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests


def load_grok_md_instructions() -> str:
    """Load GROK.md instructions from current directory and global config"""
    instructions = []

    # Check for project-specific .grok/GROK.md
    project_grok = Path.cwd() / ".grok" / "GROK.md"
    if project_grok.exists():
        try:
            with open(project_grok) as f:
                content = f.read().strip()
                if content:
                    instructions.append(f"\n# Project Instructions (from {project_grok}):\n{content}")
        except:
            pass

    # Check for global ~/.grok/GROK.md
    global_grok = Path.home() / ".grok" / "GROK.md"
    if global_grok.exists():
        try:
            with open(global_grok) as f:
                content = f.read().strip()
                if content:
                    instructions.append(f"\n# Global Instructions (from {global_grok}):\n{content}")
        except:
            pass

    return "\n".join(instructions)


SYSTEM_PROMPT = """Grok Code: AI coding assistant with tools: bash, read_file, write_file, edit_file, glob, grep, ask_user_question, spawn_task.

Rules:
- Use file tools (NOT bash) for file ops
- Always read before editing
- Match indentation exactly in edits
- Complete tasks fully (no TODOs)
- Concise responses
- ask_user_question for clarifications
- spawn_task for complex sub-tasks

Model: {current_model} | CWD: {cwd} | Platform: {platform} | Date: {date} | Session: {session_id}
{grok_md_instructions}
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


class AskUserQuestionTool(Tool):
    """Ask user questions during execution"""

    def __init__(self):
        super().__init__("AskUserQuestion")

    def execute(self, question: str) -> str:
        """Ask user a question and return their response"""
        try:
            print(f"\n\033[1;33m[Question]\033[0m {question}")
            response = input("\033[1;36m> \033[0m").strip()
            return response if response else "User provided no response"
        except (KeyboardInterrupt, EOFError):
            return "User cancelled question"


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

            # Create sub-client with same config but different model and NO parent client (prevent recursion)
            sub_client = GrokClient(
                self.parent_client.api_key,
                self.parent_client.config_dir,
                model=model,
                session_id=f"{self.parent_client.session_id}-subtask"
            )
            sub_client.verbose = self.parent_client.verbose
            sub_client.auto_select_model = False  # Sub-agents use specified model

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
            "ask_user_question": AskUserQuestionTool(),
        }

        if parent_client:
            self.tools["spawn_task"] = SpawnTaskTool(parent_client)

    def get_tool_definitions(self) -> List[Dict]:
        defs = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute bash command (git, npm, docker, tests)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout": {"type": "integer"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file with line numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "offset": {"type": "integer"},
                            "limit": {"type": "integer"}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write/overwrite file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Replace exact text in file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                            "replace_all": {"type": "boolean"}
                        },
                        "required": ["file_path", "old_string", "new_string"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "glob",
                    "description": "Find files by pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"}
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "grep",
                    "description": "Search files (regex)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"},
                            "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]},
                            "glob": {"type": "string"},
                            "case_insensitive": {"type": "boolean"}
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_user_question",
                    "description": "Ask user a question interactively",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"}
                        },
                        "required": ["question"]
                    }
                }
            }
        ]

        if self.parent_client:
            defs.append({
                "type": "function",
                "function": {
                    "name": "spawn_task",
                    "description": "Spawn sub-agent (fast/coding/reasoning model)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "model": {"type": "string", "enum": ["grok-3-mini", "grok-code-fast-1", "grok-4-fast-reasoning"]},
                            "description": {"type": "string"}
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


class SessionManager:
    """Manage conversation sessions with unique IDs"""

    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.sessions_dir = self.config_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.config_dir / "history.jsonl"  # Global log

    def generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{timestamp}_{short_uuid}"

    def get_session_file(self, session_id: str) -> Path:
        """Get path to session file"""
        return self.sessions_dir / f"{session_id}.json"

    def load_session(self, session_id: str) -> List[Dict]:
        """Load messages from a session"""
        session_file = self.get_session_file(session_id)
        if not session_file.exists():
            return []

        try:
            with open(session_file) as f:
                data = json.load(f)
                return data.get("messages", [])
        except:
            return []

    def save_session(self, session_id: str, messages: List[Dict], metadata: Dict = None):
        """Save session to file"""
        session_file = self.get_session_file(session_id)

        # Preserve existing metadata if file exists
        existing_metadata = {}
        if session_file.exists():
            try:
                with open(session_file) as f:
                    existing_data = json.load(f)
                    existing_metadata = existing_data.get("metadata", {})
            except:
                pass

        # Merge metadata
        final_metadata = {**existing_metadata, **(metadata or {})}

        data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "cwd": os.getcwd(),
            "messages": messages,
            "metadata": final_metadata
        }

        try:
            with open(session_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save session: {e}", file=sys.stderr)

    def rename_session(self, session_id: str, name: str):
        """Rename a session"""
        session_file = self.get_session_file(session_id)
        if not session_file.exists():
            return False

        try:
            with open(session_file) as f:
                data = json.load(f)

            data["metadata"]["name"] = name

            with open(session_file, 'w') as f:
                json.dump(data, f, indent=2)

            return True
        except:
            return False

    def log_to_history(self, session_id: str, message: Dict):
        """Log message to global history file"""
        try:
            with open(self.history_file, 'a') as f:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id,
                    "cwd": os.getcwd(),
                    "message": message
                }
                f.write(json.dumps(entry) + "\n")
        except:
            pass

    def list_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """List recent sessions"""
        sessions = []
        for session_file in sorted(self.sessions_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                with open(session_file) as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": data["session_id"],
                        "created_at": data.get("created_at", "unknown"),
                        "cwd": data.get("cwd", "unknown"),
                        "message_count": len(data.get("messages", [])),
                        "name": data.get("metadata", {}).get("name", "")
                    })
            except:
                continue
        return sessions


class GrokClient:
    # Grok API pricing (per 1M tokens)
    PRICING = {
        "grok-3-mini": {"input": 0.20, "output": 0.20},
        "grok-code-fast-1": {"input": 5.00, "output": 15.00},
        "grok-4-fast-reasoning": {"input": 5.00, "output": 15.00},
    }

    def __init__(self, api_key: str, config_dir: str, model: str = None, session_id: str = None):
        self.api_key = api_key
        self.model = model or ModelSelector.CODING_MODEL
        self.base_url = "https://api.x.ai/v1"
        self.config_dir = config_dir

        # Session management
        self.session_manager = SessionManager(config_dir)
        self.session_id = session_id or self.session_manager.generate_session_id()
        self.messages: List[Dict] = []
        self.session_name = None

        self.tool_registry = ToolRegistry(parent_client=self)
        self.verbose = False
        self.auto_select_model = True
        self.grok_md_instructions = load_grok_md_instructions()

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.total_cost = 0.0

    def load_session(self, session_id: str):
        """Load existing session"""
        self.session_id = session_id
        self.messages = self.session_manager.load_session(session_id)
        if self.verbose:
            print(f"Loaded session {session_id} with {len(self.messages)} messages", file=sys.stderr)

    def save_session(self):
        """Save current session"""
        metadata = {
            "model": self.model,
            "auto_select": self.auto_select_model
        }
        if self.session_name:
            metadata["name"] = self.session_name
        self.session_manager.save_session(self.session_id, self.messages, metadata)

    def get_system_message(self) -> str:
        return SYSTEM_PROMPT.format(
            current_model=self.model,
            cwd=os.getcwd(),
            platform=f"{platform.system()} {platform.release()}",
            date=datetime.now().strftime("%Y-%m-%d"),
            session_id=self.session_id,
            grok_md_instructions=self.grok_md_instructions
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

        # Add user message
        user_msg = {"role": "user", "content": user_message}
        self.messages.append(user_msg)
        self.session_manager.log_to_history(self.session_id, user_msg)

        # Build messages for API (system + conversation history)
        # Enable prompt caching on system message for 90% token savings
        messages = [
            {
                "role": "system",
                "content": self.get_system_message(),
                "cache_control": {"type": "ephemeral"}
            }
        ] + self.messages

        max_iterations = 25
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

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

        # Save session after completing chat
        self.save_session()

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
            self.messages.append(current_message)
            self.session_manager.log_to_history(self.session_id, current_message)
            messages.append(current_message)

            yield "\n"
            for tool_call in tool_calls:
                yield from self._execute_tool_call(tool_call, messages)

            return True

        if full_content:
            current_message["content"] = full_content
            self.messages.append(current_message)
            self.session_manager.log_to_history(self.session_id, current_message)
            messages.append(current_message)

        return False

    def _handle_non_stream_response(self, response, messages: List[Dict]) -> bool:
        data = response.json()
        message = data["choices"][0]["message"]

        # Track usage
        if "usage" in data:
            self._track_usage(data["usage"])

        if message.get("tool_calls"):
            self.messages.append(message)
            self.session_manager.log_to_history(self.session_id, message)
            messages.append(message)

            for tool_call in message["tool_calls"]:
                yield from self._execute_tool_call(tool_call, messages)

            return True

        if message.get("content"):
            self.messages.append(message)
            self.session_manager.log_to_history(self.session_id, message)
            messages.append(message)
            yield message["content"]

        return False

    def _track_usage(self, usage: Dict):
        """Track token usage and calculate cost"""
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cached_tokens = usage.get("cached_tokens", 0)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cached_tokens += cached_tokens

        # Calculate cost (cached tokens are free)
        uncached_input = input_tokens - cached_tokens
        pricing = self.PRICING.get(self.model, {"input": 5.00, "output": 15.00})
        cost = (uncached_input / 1_000_000 * pricing["input"]) + (output_tokens / 1_000_000 * pricing["output"])
        self.total_cost += cost

        if self.verbose:
            print(f"\n[Tokens: {input_tokens} in ({cached_tokens} cached), {output_tokens} out | Cost: ${cost:.4f}]", file=sys.stderr)

    def get_cost_summary(self) -> str:
        """Get formatted cost summary"""
        savings = (self.total_cached_tokens / max(self.total_input_tokens, 1)) * 100
        return f"""Cost Summary:
Input tokens:  {self.total_input_tokens:,} ({self.total_cached_tokens:,} cached, {savings:.1f}% savings)
Output tokens: {self.total_output_tokens:,}
Total cost:    ${self.total_cost:.4f}"""

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
        self.messages.append(tool_message)
        self.session_manager.log_to_history(self.session_id, tool_message)
        messages.append(tool_message)


def interactive_mode(client: GrokClient):
    print("Grok Code - AI Coding Assistant")
    print(f"Model: {client.model} | Session: {client.session_id[:16]}...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Auto-select: {'enabled' if client.auto_select_model else 'disabled'}")
    print("\nCommands: /clear /save /resume /rename /sessions /cost /verbose /auto /model /exit\n")

    while True:
        try:
            user_input = input("\033[1;36m>\033[0m ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['/exit', '/quit']:
                client.save_session()
                print(f"Session saved: {client.session_id}")
                break

            if user_input == '/clear':
                client.messages = []
                client.session_id = client.session_manager.generate_session_id()
                print(f"✓ New session started: {client.session_id}")
                continue

            if user_input.startswith('/save'):
                client.save_session()
                print(f"✓ Session saved: {client.session_id}")
                continue

            if user_input.startswith('/resume'):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: /resume <session-id>")
                    continue
                try:
                    client.load_session(parts[1])
                    print(f"✓ Resumed session: {parts[1]} ({len(client.messages)} messages)")
                except Exception as e:
                    print(f"✗ Failed to load session: {e}")
                continue

            if user_input.startswith('/rename'):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: /rename <name>")
                    continue
                if client.session_manager.rename_session(client.session_id, parts[1]):
                    print(f"✓ Renamed session to: {parts[1]}")
                else:
                    print("✗ Failed to rename session")
                continue

            if user_input == '/sessions':
                sessions = client.session_manager.list_recent_sessions()
                print("\nRecent sessions:")
                for s in sessions:
                    name = f" [{s['name']}]" if s['name'] else ""
                    print(f"  {s['session_id']}{name} - {s['message_count']} messages - {s['created_at']}")
                print()
                continue

            if user_input == '/cost':
                print(f"\n{client.get_cost_summary()}\n")
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
            client.save_session()
            print(f"\nSession saved: {client.session_id}")
            break
        except Exception as e:
            print(f"\nError: {e}")

    print("Goodbye!")


def main():
    import argparse

    config_dir = os.path.expanduser("~/.grok")

    parser = argparse.ArgumentParser(
        description='Grok Code - AI coding assistant with session management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  grok "create hello world"              # New session
  grok --resume <session-id> "continue"  # Resume session
  grok --sessions                        # List recent sessions
  grok --verbose "debug this"            # Verbose mode
        """
    )
    parser.add_argument('prompt', nargs='*', help='Prompt for Grok')
    parser.add_argument('--model', help='Model to use (default: auto-select grok-code-fast-1)')
    parser.add_argument('--resume', metavar='SESSION_ID', help='Resume existing session')
    parser.add_argument('-c', '--continue', dest='continue_last', action='store_true', help='Continue from most recent session')
    parser.add_argument('-p', '--print', dest='print_mode', action='store_true', help='Print mode: output only, no streaming UI')
    parser.add_argument('--name', metavar='NAME', help='Name for this session')
    parser.add_argument('--sessions', action='store_true', help='List recent sessions')
    parser.add_argument('--no-stream', action='store_true', help='Disable streaming')
    parser.add_argument('--version', action='version', version='Grok Code 3.0.0')
    parser.add_argument('--config-dir', default=config_dir, help='Config directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose tool output')
    parser.add_argument('--no-auto-select', action='store_true', help='Disable automatic model selection')

    args = parser.parse_args()

    api_key = os.environ.get('GROK_API_KEY')
    if not api_key:
        print("Error: GROK_API_KEY not set")
        print("Usage: export GROK_API_KEY='your-key'")
        sys.exit(1)

    # Handle --sessions flag
    if args.sessions:
        session_mgr = SessionManager(args.config_dir)
        sessions = session_mgr.list_recent_sessions(20)
        print("\nRecent sessions:")
        print("-" * 80)
        for s in sessions:
            name = f" [{s['name']}]" if s['name'] else ""
            print(f"{s['session_id']}{name} | {s['message_count']:3} msgs | {s['created_at']} | {s['cwd']}")
        print("-" * 80)
        print(f"\nTo resume: grok --resume <session-id>")
        sys.exit(0)

    # Create client
    client = GrokClient(api_key, args.config_dir, model=args.model)
    client.verbose = args.verbose
    client.auto_select_model = not args.no_auto_select
    client.session_name = args.name  # Store name to save later

    # Handle --continue flag (resume most recent session)
    if args.continue_last:
        session_mgr = SessionManager(args.config_dir)
        sessions = session_mgr.list_recent_sessions(1)
        if sessions:
            session_id = sessions[0]['session_id']
            try:
                client.load_session(session_id)
                if args.verbose:
                    print(f"✓ Continued session: {session_id} ({len(client.messages)} messages)\n")
            except Exception as e:
                print(f"Error: Failed to continue session: {e}")
                sys.exit(1)
        else:
            print("No previous sessions found to continue")
            sys.exit(1)

    # Resume session if requested
    elif args.resume:
        try:
            client.load_session(args.resume)
            if client.verbose:
                print(f"✓ Resumed session: {args.resume} ({len(client.messages)} messages)\n")
        except Exception as e:
            print(f"Error: Failed to load session '{args.resume}': {e}")
            sys.exit(1)

    # Check for stdin input (piped)
    stdin_input = ""
    if not sys.stdin.isatty():
        stdin_input = sys.stdin.read().strip()

    if args.prompt:
        prompt = ' '.join(args.prompt)

        # Append stdin if provided
        if stdin_input:
            prompt = f"{prompt}\n\nInput:\n{stdin_input}"

        if args.print_mode:
            # Print mode: collect all output, print without formatting
            output = []
            for chunk in client.chat(prompt, stream=False):
                output.append(chunk)
            print(''.join(output))
        else:
            # Normal mode: stream output
            for chunk in client.chat(prompt, stream=not args.no_stream):
                print(chunk, end='', flush=True)
            print()
            if args.verbose:
                print(f"\nSession ID: {client.session_id}")
    else:
        interactive_mode(client)


if __name__ == '__main__':
    main()
