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
import signal
import threading

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

import concurrent.futures
import hashlib
import time
import base64
import mimetypes


# ============================================================================
# COLOR SCHEME & UI CONSTANTS
# ============================================================================

class Colors:
    """ANSI color codes for terminal styling"""
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

    # Reset
    RESET = '\033[0m'

    # Semantic colors
    SUCCESS = BRIGHT_GREEN
    ERROR = BRIGHT_RED
    WARNING = BRIGHT_YELLOW
    INFO = BRIGHT_CYAN
    TOOL = BRIGHT_MAGENTA
    PROMPT = BRIGHT_CYAN + BOLD
    HEADER = BRIGHT_BLUE + BOLD
    MUTED = BRIGHT_BLACK


class UI:
    """UI elements and formatting"""

    # Icons
    CHECKMARK = f"{Colors.SUCCESS}✓{Colors.RESET}"
    CROSS = f"{Colors.ERROR}✗{Colors.RESET}"
    ARROW = f"{Colors.CYAN}→{Colors.RESET}"
    BULLET = f"{Colors.MUTED}•{Colors.RESET}"
    SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    # Tool icons
    TOOL_ICONS = {
        'bash': '🔧',
        'read_file': '📖',
        'write_file': '✍️ ',
        'edit_file': '✏️ ',
        'glob': '🔍',
        'grep': '🔎',
        'ask_user_question': '❓',
        'spawn_task': '🚀',
        'web_fetch': '🌐',
        'web_search': '🔎',
        'todo_write': '📋',
        'plan_mode': '🗺️ '
    }

    # Box drawing
    BOX_TOP = "┌─"
    BOX_BOTTOM = "└─"
    BOX_MIDDLE = "├─"
    BOX_VERTICAL = "│"
    BOX_HORIZONTAL = "─"

    @staticmethod
    def banner():
        """Return stylish Grok Code banner"""
        return f"""{Colors.BRIGHT_CYAN}{Colors.BOLD}
  ╔═══════════════════════════════════════════════════════╗
  ║                                                       ║
  ║   {Colors.BRIGHT_WHITE}██████╗ ██████╗  ██████╗ ██╗  ██╗{Colors.BRIGHT_CYAN}            ║
  ║   {Colors.BRIGHT_WHITE}██╔════╝ ██╔══██╗██╔═══██╗██║ ██╔╝{Colors.BRIGHT_CYAN}            ║
  ║   {Colors.BRIGHT_WHITE}██║  ███╗██████╔╝██║   ██║█████╔╝{Colors.BRIGHT_CYAN}             ║
  ║   {Colors.BRIGHT_WHITE}██║   ██║██╔══██╗██║   ██║██╔═██╗{Colors.BRIGHT_CYAN}             ║
  ║   {Colors.BRIGHT_WHITE}╚██████╔╝██║  ██║╚██████╔╝██║  ██╗{Colors.BRIGHT_CYAN}            ║
  ║   {Colors.BRIGHT_WHITE} ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝{Colors.BRIGHT_CYAN}            ║
  ║                                                       ║
  ║   {Colors.BRIGHT_MAGENTA}AI Coding Assistant v3.1{Colors.BRIGHT_CYAN}                     ║
  ║   {Colors.MUTED}Powered by xAI{Colors.BRIGHT_CYAN}                                ║
  ║                                                       ║
  ╚═══════════════════════════════════════════════════════╝
{Colors.RESET}"""

    @staticmethod
    def header(text: str, width: int = 60) -> str:
        """Create a formatted header"""
        padding = (width - len(text) - 2) // 2
        return f"\n{Colors.HEADER}{'═' * width}\n{' ' * padding} {text} {' ' * padding}\n{'═' * width}{Colors.RESET}\n"

    @staticmethod
    def separator(width: int = 60) -> str:
        """Create a separator line"""
        return f"{Colors.MUTED}{'─' * width}{Colors.RESET}"

    @staticmethod
    def success(text: str) -> str:
        """Format success message"""
        return f"{UI.CHECKMARK} {Colors.SUCCESS}{text}{Colors.RESET}"

    @staticmethod
    def error(text: str) -> str:
        """Format error message"""
        return f"{UI.CROSS} {Colors.ERROR}{text}{Colors.RESET}"

    @staticmethod
    def warning(text: str) -> str:
        """Format warning message"""
        return f"{Colors.WARNING}⚠ {text}{Colors.RESET}"

    @staticmethod
    def info(text: str) -> str:
        """Format info message"""
        return f"{Colors.INFO}ℹ {text}{Colors.RESET}"

    @staticmethod
    def tool(tool_name: str) -> str:
        """Format tool execution indicator"""
        icon = UI.TOOL_ICONS.get(tool_name, '🔨')
        return f"{Colors.TOOL}{icon} {tool_name}{Colors.RESET}"

    @staticmethod
    def prompt() -> str:
        """Get styled prompt"""
        return f"{Colors.PROMPT}❯{Colors.RESET} "


class Spinner:
    """Animated spinner for long operations"""

    def __init__(self, text: str = ""):
        self.text = text
        self.spinning = False
        self.thread = None
        self.current_frame = 0

    def _spin(self):
        """Spinner animation loop"""
        while self.spinning:
            frame = UI.SPINNER[self.current_frame % len(UI.SPINNER)]
            sys.stderr.write(f"\r{Colors.CYAN}{frame}{Colors.RESET} {self.text}...")
            sys.stderr.flush()
            self.current_frame += 1
            time.sleep(0.1)

    def start(self):
        """Start spinner"""
        if not sys.stderr.isatty():
            return
        self.spinning = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self, final_text: str = None):
        """Stop spinner and optionally show final message"""
        if not sys.stderr.isatty():
            return
        self.spinning = False
        if self.thread:
            self.thread.join(timeout=0.2)
        sys.stderr.write("\r" + " " * (len(self.text) + 10) + "\r")
        if final_text:
            sys.stderr.write(f"{final_text}\n")
        sys.stderr.flush()


# Signal handling for graceful exit
_exit_requested = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global _exit_requested
    if _exit_requested:
        # Second Ctrl+C = force exit
        print(f"\n\n{UI.warning('Force exit')}")
        sys.exit(130)
    else:
        _exit_requested = True
        print(f"\n\n{UI.info('Exiting gracefully... (Press Ctrl+C again to force)')}")
        sys.exit(0)


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


SYSTEM_PROMPT = """Grok Code: AI coding assistant with tools: bash, read_file, write_file, edit_file, glob, grep, ask_user_question, spawn_task, web_fetch, web_search, todo_write, plan_mode.

Rules:
- Use file tools (NOT bash) for file ops
- read_file detects images and returns metadata (size, type, dimensions)
- Always read before editing
- Match indentation exactly in edits
- Complete tasks fully (no TODOs)
- Concise responses
- ask_user_question for clarifications
- spawn_task for complex sub-tasks
- web_fetch to fetch web pages
- web_search for DuckDuckGo searches
- todo_write for task list management
- plan_mode for creating implementation plans

Model: {current_model} | CWD: {cwd} | Platform: {platform} | Date: {date} | Session: {session_id}
{grok_md_instructions}
"""


class ModelSelector:
    """Intelligent model selection based on task complexity"""

    # Model tier mapping (Claude Code equivalent)
    FAST_MODEL = "grok-3-mini"           # Haiku tier: simple, fast, cheap
    CODING_MODEL = "grok-code-fast-1"    # Sonnet tier: balanced coding (DEFAULT)
    REASONING_MODEL = "grok-4-fast-reasoning"  # Opus tier: complex reasoning
    VISION_MODEL = "grok-2-vision-1212"  # Vision: image analysis

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
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.ico'}

    def _is_image(self, file_path: str) -> bool:
        """Check if file is an image based on extension"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.image_extensions

    def _read_image(self, file_path: str) -> str:
        """Read image file and return metadata (vision not yet supported)"""
        try:
            with open(file_path, 'rb') as f:
                image_data = f.read()

            # Get file size
            file_size = len(image_data)
            size_kb = file_size / 1024
            size_mb = size_kb / 1024

            # Get mime type
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = 'application/octet-stream'

            # Get image dimensions if possible
            dimensions = ""
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(image_data))
                dimensions = f", {img.width}x{img.height}px"
            except:
                pass

            # Return text description
            if size_mb >= 1:
                size_str = f"{size_mb:.1f}MB"
            else:
                size_str = f"{size_kb:.1f}KB"

            return f"Image file: {file_path}\nType: {mime_type}\nSize: {size_str}{dimensions}\n\nNote: Image analysis/vision is not yet supported. This is just metadata about the image file."

        except Exception as e:
            return f"Error reading image: {str(e)}"

    def execute(self, file_path: str, offset: int = 0, limit: int = 0) -> str:
        try:
            file_path = os.path.expanduser(file_path)

            # Check if it's an image
            if self._is_image(file_path):
                return self._read_image(file_path)

            # Regular text file handling
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
            print(f"\n{Colors.YELLOW}❓ {Colors.BOLD}{question}{Colors.RESET}")
            response = input(UI.prompt()).strip()
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


class WebFetchTool(Tool):
    """Fetch and parse web pages"""

    def __init__(self):
        super().__init__("WebFetch")

    def execute(self, url: str, timeout: int = 10) -> str:
        try:
            import requests
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                return "Error: beautifulsoup4 not installed.\n\nInstall with:\n  pip install beautifulsoup4"

            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; GrokCode/1.0)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }

            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, verify=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
                element.decompose()

            text = soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)

            max_chars = 10000
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n... (truncated, total {len(text)} chars)"

            return f"URL: {url}\nStatus: {response.status_code}\nContent-Type: {response.headers.get('Content-Type', 'unknown')}\n\n{text}"

        except requests.exceptions.Timeout:
            return f"Error: Request timed out after {timeout}s for {url}"
        except requests.exceptions.SSLError as e:
            return f"Error: SSL certificate error for {url}: {str(e)}"
        except requests.exceptions.RequestException as e:
            return f"Error: Failed to fetch {url}: {str(e)}"
        except Exception as e:
            return f"Error parsing page: {str(e)}"


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo"""

    def __init__(self):
        super().__init__("WebSearch")

    def _install_package(self, package_name: str) -> bool:
        """Try to install package using various methods"""
        # Try pip with --user first (works on most systems)
        try:
            print(f"Installing {package_name}...", file=sys.stderr)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "--user", package_name],
                stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError:
            pass

        # Try system package managers
        if os.path.exists("/usr/bin/pacman"):  # Arch Linux
            try:
                pkg_name = f"python-{package_name}"
                print(f"Trying system install: sudo pacman -S --noconfirm {pkg_name}", file=sys.stderr)
                subprocess.check_call(
                    ["sudo", "pacman", "-S", "--noconfirm", pkg_name],
                    stderr=subprocess.DEVNULL
                )
                return True
            except:
                pass

        # Last resort: --break-system-packages (risky but user chose to use this)
        try:
            print(f"Trying pip install with --break-system-packages...", file=sys.stderr)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", package_name],
                stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError:
            pass

        return False

    def execute(self, query: str, num_results: int = 5) -> str:
        try:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                # Try to install
                if not self._install_package("duckduckgo-search"):
                    return (
                        "Error: Could not install duckduckgo-search library.\n\n"
                        "Please install manually:\n"
                        "  Arch Linux: sudo pacman -S python-duckduckgo-search\n"
                        "  or: pip install --user duckduckgo-search\n"
                        "  or: pip install --break-system-packages duckduckgo-search"
                    )

                # Try import again after install
                try:
                    from duckduckgo_search import DDGS
                except ImportError:
                    return "Error: Installation succeeded but import still fails. Try restarting grok."

            results = []
            with DDGS() as ddgs:
                for i, r in enumerate(ddgs.text(query, max_results=num_results)):
                    results.append(
                        f"{i+1}. {r['title']}\n"
                        f"   URL: {r['href']}\n"
                        f"   {r['body']}\n"
                    )

            if results:
                return f"Search results for '{query}':\n\n" + "\n".join(results)
            else:
                return f"No results found for '{query}'"

        except Exception as e:
            return f"Error performing search: {str(e)}"


class TodoWriteTool(Tool):
    """Task list management for planning and tracking"""

    def __init__(self):
        super().__init__("TodoWrite")
        self.todos = []

    def execute(self, todos: List[Dict]) -> str:
        try:
            self.todos = todos

            status_icons = {
                'completed': '✅',
                'in_progress': '🔄',
                'pending': '⏳'
            }

            output = ["📋 Task List:"]

            for i, todo in enumerate(todos, 1):
                status = todo.get('status', 'pending')
                content = todo.get('content', '')
                icon = status_icons.get(status, '•')
                output.append(f"{icon} {i}. {content}")

            total = len(todos)
            completed = sum(1 for t in todos if t.get('status') == 'completed')
            in_progress = sum(1 for t in todos if t.get('status') == 'in_progress')

            output.append("")
            output.append(f"Progress: {completed}/{total} completed, {in_progress} in progress")

            return "\n".join(output)

        except Exception as e:
            return f"Error updating todo list: {str(e)}"


class PlanModeTool(Tool):
    """Enter planning mode for complex tasks"""

    def __init__(self, client):
        super().__init__("PlanMode")
        self.client = client

    def execute(self, task_description: str) -> str:
        try:
            original_model = self.client.model
            self.client.model = "grok-4-fast-reasoning"

            plan_prompt = f"""Task: {task_description}

Create a detailed implementation plan following this structure:

1. UNDERSTANDING
   - What is the task asking for?
   - What are the key requirements?
   - What are the constraints?

2. APPROACH
   - What's the best way to solve this?
   - What are alternative approaches?
   - Why is this approach best?

3. IMPLEMENTATION STEPS
   - Step 1: [Concrete action]
   - Step 2: [Concrete action]
   - ...

4. FILES TO CREATE/MODIFY
   - file1.py: [Purpose]
   - file2.py: [Purpose]
   - ...

5. TESTING STRATEGY
   - How will we verify it works?
   - What edge cases to test?

6. POTENTIAL RISKS
   - What could go wrong?
   - How to mitigate?

Provide a clear, actionable plan."""

            plan_parts = []
            for chunk in self.client.chat(plan_prompt, stream=False):
                plan_parts.append(chunk)

            plan = ''.join(plan_parts)
            self.client.model = original_model

            return f"📋 IMPLEMENTATION PLAN\n\n{plan}\n\n{'='*60}\nReady to implement? Ask me to proceed."

        except Exception as e:
            self.client.model = original_model
            return f"Error creating plan: {str(e)}"


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
            "web_fetch": WebFetchTool(),
            "web_search": WebSearchTool(),
            "todo_write": TodoWriteTool(),
        }

        if parent_client:
            self.tools["spawn_task"] = SpawnTaskTool(parent_client)
            self.tools["plan_mode"] = PlanModeTool(parent_client)

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
                    "description": "Read file with line numbers. Detects images (png, jpg, gif, etc) and returns metadata (size, type, dimensions).",
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

        defs.extend([
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "Fetch and parse web page content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to fetch"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds"}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search web using DuckDuckGo",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "num_results": {"type": "integer", "description": "Number of results"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "todo_write",
                    "description": "Update task list for planning",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "todos": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                                        "activeForm": {"type": "string"}
                                    },
                                    "required": ["content", "status", "activeForm"]
                                }
                            }
                        },
                        "required": ["todos"]
                    }
                }
            }
        ])

        if self.parent_client:
            defs.extend([
                {
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
                },
                {
                    "type": "function",
                    "function": {
                        "name": "plan_mode",
                        "description": "Create implementation plan for complex tasks",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_description": {"type": "string"}
                            },
                            "required": ["task_description"]
                        }
                    }
                }
            ])

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

    def list_recent_sessions(self, limit: int = 10, cwd_filter: str = None) -> List[Dict]:
        """List recent sessions, optionally filtered by working directory"""
        sessions = []
        for session_file in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            try:
                with open(session_file) as f:
                    data = json.load(f)
                    session_cwd = data.get("cwd", "")

                    # Filter by cwd if requested
                    if cwd_filter and session_cwd != cwd_filter:
                        continue

                    sessions.append({
                        "session_id": data["session_id"],
                        "created_at": data.get("created_at", "unknown"),
                        "cwd": session_cwd,
                        "message_count": len(data.get("messages", [])),
                        "name": data.get("metadata", {}).get("name", "")
                    })

                    if len(sessions) >= limit:
                        break
            except:
                continue
        return sessions


class GrokClient:
    # Grok API pricing (per 1M tokens)
    PRICING = {
        "grok-3-mini": {"input": 0.20, "output": 0.20},
        "grok-code-fast-1": {"input": 5.00, "output": 15.00},
        "grok-4-fast-reasoning": {"input": 5.00, "output": 15.00},
        "grok-2-vision-1212": {"input": 2.00, "output": 10.00},
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

        # Speed optimization: Connection pooling
        self.http_session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20,
            pool_block=False
        )
        self.http_session.mount("https://", adapter)
        self.http_session.mount("http://", adapter)
        self.http_session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

        # Speed optimization: Response caching
        self.response_cache = {}
        self.cache_enabled = True
        self.cache_max_size = 100

        # Max iteration time for API calls
        self.max_iteration_time = 300  # 5 minutes

        # Parallel execution safe tools
        self.PARALLEL_SAFE_TOOLS = {'read_file', 'glob', 'grep', 'bash'}

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


    def _cache_key(self, messages: List[Dict], model: str) -> str:
        """Generate cache key from messages and model"""
        cache_content = {
            'model': model,
            'messages': [
                {'role': msg['role'], 'content': msg.get('content', '')}
                for msg in messages
                if msg['role'] in ['user', 'assistant']
            ]
        }
        content_str = json.dumps(cache_content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()

    def _can_run_parallel(self, tool_name: str, arguments: Dict) -> bool:
        """Check if tool can safely run in parallel"""
        if tool_name not in self.PARALLEL_SAFE_TOOLS:
            return False

        if tool_name == 'bash':
            command = arguments.get('command', '').lower()
            safe_prefixes = ['ls', 'cat', 'head', 'tail', 'grep', 'find',
                           'echo', 'pwd', 'which', 'wc', 'diff', 'git status',
                           'git log', 'git diff', 'npm list', 'pip list']
            return any(command.startswith(prefix) for prefix in safe_prefixes)

        return True

    def _execute_tool_calls_parallel(self, tool_calls: List[Dict], messages: List[Dict]) -> List:
        """Execute tool calls in parallel where safe"""
        parallel_calls = []
        sequential_calls = []

        for tc in tool_calls:
            tool_name = tc['function']['name']
            try:
                arguments = json.loads(tc['function']['arguments'])
            except:
                arguments = {}

            if self._can_run_parallel(tool_name, arguments):
                parallel_calls.append((tc, arguments))
            else:
                sequential_calls.append((tc, arguments))

        all_results = []

        # Execute parallel calls concurrently
        if parallel_calls:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_call = {
                    executor.submit(self.tool_registry.execute, tc['function']['name'], args): tc
                    for tc, args in parallel_calls
                }

                for future in concurrent.futures.as_completed(future_to_call):
                    tc = future_to_call[future]
                    try:
                        result = future.result()
                        all_results.append((tc, result))
                    except Exception as e:
                        all_results.append((tc, f"Error: {str(e)}"))

        # Execute sequential calls one by one
        for tc, args in sequential_calls:
            result = self.tool_registry.execute(tc['function']['name'], args)
            all_results.append((tc, result))

        # Add all results to messages
        for tc, result in all_results:
            self._add_tool_result(tc["id"], result, messages)

        return all_results

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


        max_iterations = 50  # Increased from 25
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            if iteration >= max_iterations - 3 and self.verbose:
                print(f"\n[Warning: {max_iterations - iteration} iterations remaining]", file=sys.stderr)

            payload = {
                "model": working_model,
                "messages": messages,
                "tools": self.tool_registry.get_tool_definitions(),
                "stream": stream,
                "temperature": 0.7
            }

            try:
                response = self.http_session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    stream=stream,
                    timeout=self.max_iteration_time
                )
                response.raise_for_status()
            except requests.exceptions.Timeout:
                yield f"\n⚠️ Request timed out after {self.max_iteration_time}s.\n"
                yield "This likely means:\n"
                yield "1. Network connectivity issue\n"
                yield "2. API is down or overloaded\n"
                yield "3. Task is genuinely too complex\n"
                return
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
            yield f"\n{UI.tool(tool_name)}\n"

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
    print(UI.banner())
    print(f"{Colors.MUTED}Model:{Colors.RESET} {Colors.CYAN}{client.model}{Colors.RESET} {Colors.MUTED}|{Colors.RESET} {Colors.MUTED}Session:{Colors.RESET} {Colors.CYAN}{client.session_id[:16]}...{Colors.RESET}")
    print(f"{Colors.MUTED}Directory:{Colors.RESET} {Colors.CYAN}{os.getcwd()}{Colors.RESET}")
    print(f"{Colors.MUTED}Auto-select:{Colors.RESET} {Colors.GREEN if client.auto_select_model else Colors.RED}{'enabled' if client.auto_select_model else 'disabled'}{Colors.RESET}")
    print(f"\n{Colors.MUTED}Commands: /clear /save /resume /rename /sessions [all] /cost /verbose /auto /model /undo /retry /history /search /plan /exit{Colors.RESET}\n")

    while True:
        try:
            user_input = input(UI.prompt()).strip()

            if not user_input:
                continue

            if user_input.lower() in ['/exit', '/quit']:
                client.save_session()
                print(UI.success(f"Session saved: {client.session_id}"))
                break

            if user_input == '/clear':
                client.messages = []
                client.session_id = client.session_manager.generate_session_id()
                print(UI.success(f"New session started: {client.session_id}"))
                continue

            if user_input.startswith('/save'):
                client.save_session()
                print(UI.success(f"Session saved: {client.session_id}"))
                continue

            if user_input.startswith('/resume'):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: /resume <session-id>")
                    continue
                try:
                    client.load_session(parts[1])
                    print(UI.success(f"Resumed session: {parts[1]} ({len(client.messages)} messages)"))
                except Exception as e:
                    print(UI.error(f"Failed to load session: {e}"))
                continue

            if user_input.startswith('/rename'):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: /rename <name>")
                    continue
                if client.session_manager.rename_session(client.session_id, parts[1]):
                    print(UI.success(f"Renamed session to: {parts[1]}"))
                else:
                    print(UI.error("Failed to rename session"))
                continue

            if user_input.startswith('/sessions'):
                parts = user_input.split()
                show_all = len(parts) > 1 and parts[1] == 'all'

                if show_all:
                    sessions = client.session_manager.list_recent_sessions()
                    print("\nAll recent sessions:")
                else:
                    current_dir = os.getcwd()
                    sessions = client.session_manager.list_recent_sessions(cwd_filter=current_dir)
                    print(f"\nSessions for {current_dir}:")

                if not sessions:
                    print("  No sessions found")
                else:
                    for s in sessions:
                        name = f" [{s['name']}]" if s['name'] else ""
                        cwd_info = f" - {s['cwd']}" if show_all else ""
                        print(f"  {s['session_id']}{name} - {s['message_count']} messages - {s['created_at']}{cwd_info}")
                print()
                continue

            if user_input == '/cost':
                print(f"\n{client.get_cost_summary()}\n")
                continue

            if user_input.startswith('/verbose'):
                client.verbose = not client.verbose
                print(UI.success(f"Verbose mode: {'on' if client.verbose else 'off'}"))
                continue

            if user_input.startswith('/auto'):
                client.auto_select_model = not client.auto_select_model
                print(UI.success(f"Auto model selection: {'enabled' if client.auto_select_model else 'disabled'}"))
                continue

            if user_input.startswith('/model'):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print(f"Current model: {client.model}")
                    print("\nAvailable models:")
                    print("  grok-3-mini           - Fast, cheap")
                    print("  grok-code-fast-1      - Coding (default)")
                    print("  grok-2-vision-1212    - Vision/image analysis")
                    print("  grok-3                - Balanced")
                    print("  grok-4-fast-reasoning - Advanced reasoning")
                    continue
                client.model = parts[1]
                print(UI.success(f"Model set to: {client.model}"))
                continue

            if user_input == '/undo':
                if len(client.messages) >= 2:
                    last_user_idx = None
                    for i in range(len(client.messages) - 1, -1, -1):
                        if client.messages[i]['role'] == 'user':
                            last_user_idx = i
                            break

                    if last_user_idx is not None:
                        removed = len(client.messages) - last_user_idx
                        client.messages = client.messages[:last_user_idx]
                        print(UI.success(f"Undid last exchange ({removed} messages removed)"))
                    else:
                        print(UI.error("No user messages to undo"))
                else:
                    print(UI.error("No messages to undo"))
                continue

            if user_input.startswith('/retry'):
                parts = user_input.split()
                last_user_msg = None
                last_user_idx = None
                for i in range(len(client.messages) - 1, -1, -1):
                    if client.messages[i]['role'] == 'user':
                        last_user_msg = client.messages[i]
                        last_user_idx = i
                        break

                if not last_user_msg:
                    print("✗ No message to retry")
                    continue

                original_model = client.model
                if len(parts) > 1:
                    client.model = parts[1]
                    print(f"Retrying with model: {client.model}")

                client.messages = client.messages[:last_user_idx]

                print()
                for chunk in client.chat(last_user_msg['content'], stream=True):
                    print(chunk, end='', flush=True)
                print()

                client.model = original_model
                continue

            if user_input == '/history':
                print("\nConversation History:")
                print("-" * 60)
                for i, msg in enumerate(client.messages):
                    role = msg['role']
                    content = msg.get('content', '')

                    if role == 'user':
                        icon = '👤'
                    elif role == 'assistant':
                        icon = '🤖'
                    elif role == 'tool':
                        icon = '🔧'
                    else:
                        icon = '•'

                    if len(content) > 100:
                        content = content[:100] + "..."

                    print(f"{i+1}. {icon} {role}: {content}")

                print("-" * 60)
                print(f"Total: {len(client.messages)} messages")
                print()
                continue

            if user_input.startswith('/search '):
                query = user_input[8:].strip()
                if not query:
                    print("Usage: /search <query>")
                    continue

                print(f"\nSearching for: {query}\n")
                search_tool = WebSearchTool()
                results = search_tool.execute(query)
                print(results)
                print()
                continue

            if user_input.startswith('/plan '):
                task = user_input[6:].strip()
                if not task:
                    print("Usage: /plan <task description>")
                    continue

                print(f"\nCreating plan for: {task}\n")
                plan_tool = PlanModeTool(client)
                plan = plan_tool.execute(task)
                print(plan)
                print()
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

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    config_dir = os.path.expanduser("~/.grok")

    parser = argparse.ArgumentParser(
        description='Grok Code - AI coding assistant with session management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  grok "create hello world"              # New session
  grok --resume                          # List sessions for current directory
  grok --resume <session-id> "continue"  # Resume specific session
  grok --sessions                        # List all recent sessions
  grok --verbose "debug this"            # Verbose mode
        """
    )
    parser.add_argument('prompt', nargs='*', help='Prompt for Grok')
    parser.add_argument('--model', help='Model to use (default: auto-select grok-code-fast-1)')
    parser.add_argument('--resume', nargs='?', const='', metavar='SESSION_ID', help='Resume session (list current dir sessions if no ID given)')
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
    elif args.resume is not None:
        # If --resume provided without session ID, list sessions for current directory
        if args.resume == '':
            session_mgr = SessionManager(args.config_dir)
            current_dir = os.getcwd()
            sessions = session_mgr.list_recent_sessions(limit=20, cwd_filter=current_dir)

            if not sessions:
                print(f"\nNo sessions found for current directory: {current_dir}")
                sys.exit(0)

            print(f"\nSessions for {current_dir}:")
            print("-" * 80)
            for i, s in enumerate(sessions, 1):
                name = f" [{s['name']}]" if s['name'] else ""
                print(f"{i}. {s['session_id']}{name} | {s['message_count']:3} msgs | {s['created_at']}")
            print("-" * 80)
            print(f"\nTo resume: grok --resume <session-id>")
            sys.exit(0)
        else:
            # Resume specific session
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
