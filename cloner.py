#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TELEGRAM BOT CLONE PROXY v4.0 - ULTIMATE BEHAVIORAL MIRROR (HTML FIXED)
Author: BLACKHAT-2026
"""

# ============================================================================
# ALL IMPORTS
# ============================================================================

import asyncio
import json
import sqlite3
import re
import hashlib
import random
import time
import logging
import os
import sys
import threading
import uuid
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

from aiogram import Bot, Dispatcher, types, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    FSInputFile
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.utils.chat_action import ChatActionSender
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "4.0.0-HTML-FIXED"
BOT_TOKEN = "8653501255:AAGOwfrDxKYa3aHxWAu_FA915SAPtlotqhw"

CONFIG_DIR = Path("clone_data")
CONFIG_DIR.mkdir(exist_ok=True)
DB_PATH = CONFIG_DIR / "clone_db.sqlite"
LOGS_DIR = CONFIG_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f'bot_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    def __init__(self):
        try:
            self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self._create_tables()
            logger.info("✅ Database connected successfully")
        except Exception as e:
            logger.error(f"❌ Database connection error: {e}")
            raise
    
    def _create_tables(self):
        try:
            c = self.conn.cursor()
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS target_bots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_interactions INTEGER DEFAULT 0,
                    last_active TIMESTAMP
                )
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_uuid TEXT UNIQUE NOT NULL,
                    user_id INTEGER NOT NULL,
                    target_bot_id INTEGER NOT NULL,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    interactions INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    target_bot_id INTEGER NOT NULL,
                    direction TEXT,
                    message_type TEXT,
                    raw_data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.conn.commit()
            logger.info("✅ Database tables created/verified")
            
        except Exception as e:
            logger.error(f"❌ Error creating tables: {e}")
            raise
    
    def add_target_bot(self, username: str) -> Optional[int]:
        try:
            # Prüfen ob Bot existiert
            c = self.conn.execute(
                "SELECT id FROM target_bots WHERE username = ?", 
                (username,)
            )
            existing = c.fetchone()
            
            if existing:
                return existing[0]
            
            # Neuen Bot einfügen
            now = datetime.now().isoformat()
            c = self.conn.execute(
                "INSERT INTO target_bots (username, last_active) VALUES (?, ?)",
                (username, now)
            )
            self.conn.commit()
            
            return c.lastrowid
            
        except Exception as e:
            logger.error(f"❌ Error adding bot: {e}")
            return None
    
    def get_target_bots(self):
        try:
            c = self.conn.execute(
                "SELECT id, username, total_interactions, last_active, added_date FROM target_bots ORDER BY last_active DESC"
            )
            return [dict(row) for row in c.fetchall()]
        except Exception as e:
            logger.error(f"❌ Error getting bots: {e}")
            return []
    
    def get_target_bot(self, bot_id: int):
        try:
            c = self.conn.execute("SELECT * FROM target_bots WHERE id = ?", (bot_id,))
            row = c.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"❌ Error getting bot {bot_id}: {e}")
            return None
    
    def create_session(self, user_id: int, target_bot_id: int) -> Optional[str]:
        try:
            session_uuid = str(uuid.uuid4())
            self.conn.execute(
                "INSERT INTO sessions (session_uuid, user_id, target_bot_id, status) VALUES (?, ?, ?, ?)",
                (session_uuid, user_id, target_bot_id, "active")
            )
            self.conn.commit()
            return session_uuid
        except Exception as e:
            logger.error(f"❌ Error creating session: {e}")
            return None
    
    def end_session(self, session_uuid: str):
        try:
            self.conn.execute(
                "UPDATE sessions SET end_time = ?, status = 'ended' WHERE session_uuid = ?",
                (datetime.now().isoformat(), session_uuid)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"❌ Error ending session: {e}")
    
    def add_interaction(self, data: Dict):
        try:
            self.conn.execute(
                """INSERT INTO interactions 
                   (session_id, target_bot_id, direction, message_type, raw_data)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    data.get("session_id"),
                    data.get("target_bot_id"),
                    data.get("direction"),
                    data.get("message_type"),
                    json.dumps(data.get("raw_data", {}))
                )
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"❌ Error adding interaction: {e}")

# ============================================================================
# BOT SETUP
# ============================================================================

class BotStates(StatesGroup):
    main_menu = State()
    adding_target = State()
    cloning_session = State()

storage = MemoryStorage()
dp = Dispatcher(storage=storage)
router = Router()
db = DatabaseManager()

# Global state
active_sessions: Dict[int, Dict] = {}

# ============================================================================
# UI COMPONENTS
# ============================================================================

class UIComponents:
    @staticmethod
    def main_menu() -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        builder.row(
            InlineKeyboardButton(text="🎯 Add Target Bot", callback_data="add_target"),
            InlineKeyboardButton(text="📋 List Bots", callback_data="list_bots")
        )
        builder.row(
            InlineKeyboardButton(text="▶️ Start Clone Session", callback_data="start_clone_menu"),
        )
        builder.row(
            InlineKeyboardButton(text="ℹ️ Help", callback_data="help"),
        )
        return builder.as_markup()
    
    @staticmethod
    def bot_list(bots: List[Dict]) -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        for bot in bots:
            builder.row(InlineKeyboardButton(
                text=f"🤖 @{bot['username']} ({bot['total_interactions']} msgs)",
                callback_data=f"select_bot_{bot['id']}"
            ))
        builder.row(InlineKeyboardButton(text="🏠 Main Menu", callback_data="main_menu"))
        return builder.as_markup()
    
    @staticmethod
    def bot_actions(bot_id: int, username: str) -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        builder.row(
            InlineKeyboardButton(text="▶️ Start Clone Session", callback_data=f"clone_now_{bot_id}"),
        )
        builder.row(
            InlineKeyboardButton(text="🔙 Back to List", callback_data="list_bots"),
            InlineKeyboardButton(text="🏠 Main Menu", callback_data="main_menu")
        )
        return builder.as_markup()
    
    @staticmethod
    def session_controls() -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        builder.row(
            InlineKeyboardButton(text="⏹️ Stop Session", callback_data="stop_session"),
        )
        builder.row(
            InlineKeyboardButton(text="🏠 Main Menu", callback_data="main_menu")
        )
        return builder.as_markup()
    
    @staticmethod
    def confirm_warning() -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        builder.row(
            InlineKeyboardButton(text="✅ I UNDERSTAND", callback_data="accept_warning")
        )
        return builder.as_markup()
    
    @staticmethod
    def back_button() -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        builder.row(InlineKeyboardButton(text="🔙 Back to Main Menu", callback_data="main_menu"))
        return builder.as_markup()

# ============================================================================
# HANDLERS - ALLE MIT HTML FORMATIERUNG
# ============================================================================

@router.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext):
    """Start command"""
    logger.info(f"👤 User {message.from_user.id} started the bot")
    
    user_data = await state.get_data()
    if not user_data.get("accepted_warning", False):
        warning_text = (
            "⚠️ <b>LEGAL WARNING</b> ⚠️\n\n"
            "This tool is for <b>EDUCATIONAL PURPOSES</b> and <b>AUTHORIZED TESTING ONLY</b>.\n\n"
            "By clicking I UNDERSTAND, you confirm:\n"
            "✅ You have EXPLICIT PERMISSION\n"
            "✅ You are ONLY testing your OWN bots\n"
            "✅ You accept FULL LEGAL RESPONSIBILITY"
        )
        
        await message.answer(
            warning_text,
            parse_mode="HTML",
            reply_markup=UIComponents.confirm_warning()
        )
    else:
        await show_main_menu(message, state)

async def show_main_menu(message: types.Message, state: FSMContext):
    """Show main menu"""
    await state.set_state(BotStates.main_menu)
    await message.answer(
        f"🤖 <b>Bot Clone Proxy v{VERSION}</b>\n\n"
        f"Select an option below:",
        parse_mode="HTML",
        reply_markup=UIComponents.main_menu()
    )

@router.callback_query(F.data == "accept_warning")
async def accept_warning(callback: CallbackQuery, state: FSMContext):
    """Accept warning"""
    logger.info(f"✅ User {callback.from_user.id} accepted warning")
    await state.update_data(accepted_warning=True)
    await callback.message.delete()
    await show_main_menu(callback.message, state)
    await callback.answer()

@router.callback_query(F.data == "main_menu")
async def go_main_menu(callback: CallbackQuery, state: FSMContext):
    """Go to main menu"""
    logger.info(f"👤 User {callback.from_user.id} returned to main menu")
    await state.set_state(BotStates.main_menu)
    await callback.message.edit_text(
        f"🤖 <b>Bot Clone Proxy v{VERSION}</b>\n\n"
        f"Select an option below:",
        parse_mode="HTML",
        reply_markup=UIComponents.main_menu()
    )
    await callback.answer()

@router.callback_query(F.data == "help")
async def show_help(callback: CallbackQuery):
    """Show help"""
    await callback.message.edit_text(
        "📚 <b>Help</b>\n\n"
        "<b>How to use:</b>\n"
        "1️⃣ Add a target bot using 'Add Target Bot'\n"
        "2️⃣ Select a bot from the list\n"
        "3️⃣ Click 'Start Clone Session' to begin\n"
        "4️⃣ Send messages - they'll be forwarded to the target\n"
        "5️⃣ Responses from target will appear here\n\n"
        "<b>Features:</b>\n"
        "• Forward text, photos, videos\n"
        "• Automatic session management\n"
        "• Database storage for all interactions",
        parse_mode="HTML",
        reply_markup=UIComponents.back_button()
    )
    await callback.answer()

# ============================================================================
# ADD TARGET HANDLER - MIT HTML
# ============================================================================

@router.callback_query(F.data == "add_target")
async def add_target_prompt(callback: CallbackQuery, state: FSMContext):
    """Prompt to add target bot"""
    logger.info(f"👤 User {callback.from_user.id} clicking add_target")
    
    await state.set_state(BotStates.adding_target)
    
    text = (
        "📝 <b>Add Target Bot</b>\n\n"
        "Please send me the bot's username:\n\n"
        "Examples:\n"
        "• <code>@example_bot</code>\n"
        "• <code>example_bot</code>\n\n"
        "<i>(without the @ is also fine)</i>"
    )
    
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="🔙 Cancel", callback_data="main_menu"))
    
    await callback.message.edit_text(
        text,
        parse_mode="HTML",
        reply_markup=builder.as_markup()
    )
    await callback.answer()

@router.message(BotStates.adding_target)
async def process_add_target(message: types.Message, state: FSMContext):
    """Process new target bot - MIT HTML"""
    logger.info(f"📝 User {message.from_user.id} sending username: {message.text}")
    
    try:
        username = message.text.strip().replace('@', '').lower()
        
        if not re.match(r'^[a-zA-Z0-9_]{5,32}$', username):
            await message.answer(
                "❌ <b>Invalid username!</b>\n\n"
                "Requirements:\n"
                "• 5-32 characters\n"
                "• Only letters (a-z), numbers (0-9), and underscores (_)\n"
                "• No spaces or special characters",
                parse_mode="HTML",
                reply_markup=UIComponents.back_button()
            )
            return
        
        bot_id = db.add_target_bot(username)
        
        if bot_id:
            bot_info = db.get_target_bot(bot_id)
            added_date = bot_info.get('added_date', '')[:10] if bot_info.get('added_date') else 'Unknown'
            
            await message.answer(
                f"✅ <b>Bot @{username} added successfully!</b>\n\n"
                f"Bot ID: <code>{bot_id}</code>\n"
                f"Added: {added_date}",
                parse_mode="HTML",
                reply_markup=UIComponents.main_menu()
            )
        else:
            await message.answer(
                "❌ <b>Failed to add bot!</b>\n\n"
                "Please try again later.",
                parse_mode="HTML",
                reply_markup=UIComponents.main_menu()
            )
        
        await state.set_state(BotStates.main_menu)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        await message.answer(
            f"❌ <b>Error:</b> {str(e)[:100]}",
            parse_mode="HTML",
            reply_markup=UIComponents.main_menu()
        )
        await state.set_state(BotStates.main_menu)

# ============================================================================
# LIST BOTS HANDLER
# ============================================================================

@router.callback_query(F.data == "list_bots")
async def list_bots(callback: CallbackQuery):
    """List all target bots"""
    logger.info(f"👤 User {callback.from_user.id} listing bots")
    
    try:
        bots = db.get_target_bots()
        
        if not bots:
            await callback.message.edit_text(
                "📭 <b>No bots added yet.</b>\n\n"
                "Use 'Add Target Bot' to get started.",
                parse_mode="HTML",
                reply_markup=UIComponents.back_button()
            )
            await callback.answer()
            return
        
        await callback.message.edit_text(
            "📋 <b>Your Target Bots</b>\n\n"
            "Select a bot to manage:",
            parse_mode="HTML",
            reply_markup=UIComponents.bot_list(bots)
        )
        await callback.answer()
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        await callback.message.edit_text(
            f"❌ <b>Error:</b> {str(e)[:100]}",
            parse_mode="HTML",
            reply_markup=UIComponents.main_menu()
        )
        await callback.answer()

# ============================================================================
# START CLONE MENU HANDLER
# ============================================================================

@router.callback_query(F.data == "start_clone_menu")
async def start_clone_menu(callback: CallbackQuery):
    """Show bots to clone"""
    logger.info(f"👤 User {callback.from_user.id} opening clone menu")
    
    try:
        bots = db.get_target_bots()
        
        if not bots:
            await callback.message.edit_text(
                "❌ <b>No bots available!</b>\n\n"
                "Please add a target bot first.",
                parse_mode="HTML",
                reply_markup=UIComponents.back_button()
            )
            await callback.answer()
            return
        
        await callback.message.edit_text(
            "🎯 <b>Select a bot to clone:</b>",
            parse_mode="HTML",
            reply_markup=UIComponents.bot_list(bots)
        )
        await callback.answer()
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        await callback.message.edit_text(
            f"❌ <b>Error:</b> {str(e)[:100]}",
            parse_mode="HTML",
            reply_markup=UIComponents.main_menu()
        )
        await callback.answer()

# ============================================================================
# SELECT BOT HANDLER
# ============================================================================

@router.callback_query(F.data.startswith("select_bot_"))
async def select_bot(callback: CallbackQuery, state: FSMContext):
    """Select a bot"""
    try:
        bot_id = int(callback.data.split("_")[2])
        logger.info(f"👤 User {callback.from_user.id} selected bot {bot_id}")
        
        bot_info = db.get_target_bot(bot_id)
        
        if not bot_info:
            await callback.answer("Bot not found!", show_alert=True)
            return
        
        await state.update_data(selected_bot_id=bot_id, selected_bot_username=bot_info['username'])
        
        await callback.message.edit_text(
            f"🤖 <b>@{bot_info['username']}</b>\n\n"
            f"<b>Statistics:</b>\n"
            f"• Interactions: {bot_info['total_interactions']}\n"
            f"• Added: {bot_info['added_date'][:10] if bot_info.get('added_date') else 'Unknown'}\n\n"
            f"Choose an action:",
            parse_mode="HTML",
            reply_markup=UIComponents.bot_actions(bot_id, bot_info['username'])
        )
        await callback.answer()
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        await callback.answer("Error!", show_alert=True)

# ============================================================================
# CLONE SESSION STARTER
# ============================================================================

@router.callback_query(F.data.startswith("clone_now_"))
async def clone_now(callback: CallbackQuery, state: FSMContext):
    """START CLONE SESSION"""
    logger.info(f"🔴 User {callback.from_user.id} starting clone session")
    
    try:
        bot_id = int(callback.data.split("_")[2])
        bot_info = db.get_target_bot(bot_id)
        
        if not bot_info:
            await callback.answer("Bot not found!", show_alert=True)
            return
        
        username = bot_info['username']
        
        session_uuid = db.create_session(callback.from_user.id, bot_id)
        if not session_uuid:
            await callback.answer("Failed to create session!", show_alert=True)
            return
        
        active_sessions[callback.from_user.id] = {
            "session_uuid": session_uuid,
            "target_bot_id": bot_id,
            "target_username": username,
            "start_time": datetime.now(),
            "interactions": 0
        }
        
        await state.set_state(BotStates.cloning_session)
        
        success_text = (
            f"✅ <b>CLONE SESSION STARTED!</b>\n\n"
            f"<b>Target:</b> @{username}\n"
            f"<b>Session ID:</b> <code>{session_uuid[:8]}...</code>\n\n"
            f"<b>Instructions:</b>\n"
            f"📤 Send any message - it will be forwarded to @{username}\n"
            f"📥 Responses will appear here automatically\n"
            f"⏹️ Click 'Stop Session' when done\n\n"
            f"<b>Current Status:</b>\n"
            f"• Messages sent: 0\n"
            f"• Session active"
        )
        
        await callback.message.edit_text(
            success_text,
            parse_mode="HTML",
            reply_markup=UIComponents.session_controls()
        )
        
        await callback.answer("✅ Session started!")
        logger.info(f"✅ Session started for user {callback.from_user.id}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        await callback.message.edit_text(
            f"❌ <b>Error:</b> {str(e)[:100]}",
            parse_mode="HTML",
            reply_markup=UIComponents.main_menu()
        )
        await callback.answer("Error!", show_alert=True)

# ============================================================================
# STOP SESSION HANDLER
# ============================================================================

@router.callback_query(F.data == "stop_session")
async def stop_session(callback: CallbackQuery, state: FSMContext):
    """Stop clone session"""
    logger.info(f"👤 User {callback.from_user.id} stopping session")
    
    try:
        if callback.from_user.id in active_sessions:
            session_info = active_sessions.pop(callback.from_user.id)
            db.end_session(session_info["session_uuid"])
            
            duration = datetime.now() - session_info["start_time"]
            minutes = duration.seconds // 60
            seconds = duration.seconds % 60
            
            await callback.message.edit_text(
                f"⏹️ <b>Session Stopped!</b>\n\n"
                f"<b>Statistics:</b>\n"
                f"• Duration: {minutes}m {seconds}s\n"
                f"• Messages sent: {session_info['interactions']}\n\n"
                f"Data saved to database.",
                parse_mode="HTML",
                reply_markup=UIComponents.main_menu()
            )
        else:
            await callback.message.edit_text(
                "❌ <b>No active session found.</b>",
                parse_mode="HTML",
                reply_markup=UIComponents.main_menu()
            )
        
        await state.set_state(BotStates.main_menu)
        await callback.answer()
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        await callback.answer("Error!", show_alert=True)

# ============================================================================
# MESSAGE HANDLER DURING CLONE SESSION
# ============================================================================

@router.message(BotStates.cloning_session)
async def handle_clone_message(message: types.Message, state: FSMContext, bot: Bot):
    """Forward messages to target bot"""
    session_info = active_sessions.get(message.from_user.id)
    
    if not session_info:
        await message.answer(
            "No active session. Use /start to begin.",
            reply_markup=UIComponents.main_menu()
        )
        await state.set_state(BotStates.main_menu)
        return
    
    try:
        target_bot = session_info["target_username"]
        target_bot_id = session_info["target_bot_id"]
        
        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
        await asyncio.sleep(0.5)
        
        if message.text:
            await bot.send_message(
                chat_id=f"@{target_bot}",
                text=message.text
            )
            
            db.add_interaction({
                "session_id": session_info["session_uuid"],
                "target_bot_id": target_bot_id,
                "direction": "user_to_target",
                "message_type": "text",
                "raw_data": {"text": message.text}
            })
            
            session_info["interactions"] += 1
            
            db.conn.execute(
                "UPDATE target_bots SET total_interactions = total_interactions + 1, last_active = ? WHERE id = ?",
                (datetime.now().isoformat(), target_bot_id)
            )
            db.conn.commit()
            
            await message.reply(f"✅ Forwarded to @{target_bot}")
            
        elif message.photo:
            photo = message.photo[-1]
            await bot.send_photo(chat_id=f"@{target_bot}", photo=photo.file_id)
            await message.reply(f"✅ Photo forwarded to @{target_bot}")
            session_info["interactions"] += 1
            
        else:
            await message.reply(f"⚠️ {message.content_type} forwarding not supported")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        await message.reply(f"❌ Error: {str(e)[:100]}")

# ============================================================================
# INCOMING MESSAGES HANDLER
# ============================================================================

@router.message()
async def handle_incoming(message: types.Message, bot: Bot):
    """Handle messages from target bots"""
    try:
        if message.from_user and message.from_user.is_bot:
            bot_username = message.from_user.username
            logger.info(f"📨 Received from bot: @{bot_username}")
            
            for user_id, session_info in list(active_sessions.items()):
                if session_info["target_username"] == bot_username:
                    response = f"🤖 <b>Response from @{bot_username}</b>\n\n"
                    
                    if message.text:
                        response += message.text
                    elif message.caption:
                        response += message.caption
                    else:
                        response += f"📨 {message.content_type}"
                    
                    try:
                        if message.photo:
                            await bot.send_photo(
                                chat_id=user_id,
                                photo=message.photo[-1].file_id,
                                caption=response,
                                parse_mode="HTML"
                            )
                        elif message.video:
                            await bot.send_video(
                                chat_id=user_id,
                                video=message.video.file_id,
                                caption=response,
                                parse_mode="HTML"
                            )
                        else:
                            await bot.send_message(
                                chat_id=user_id,
                                text=response,
                                parse_mode="HTML"
                            )
                        
                        db.add_interaction({
                            "session_id": session_info["session_uuid"],
                            "target_bot_id": session_info["target_bot_id"],
                            "direction": "target_to_user",
                            "message_type": message.content_type,
                            "raw_data": {"text": message.text or message.caption}
                        })
                        
                    except Exception as e:
                        logger.error(f"❌ Error forwarding: {e}")
                    
                    break
                    
    except Exception as e:
        logger.error(f"❌ Error in incoming handler: {e}")

# ============================================================================
# ERROR HANDLER
# ============================================================================

@router.errors()
async def error_handler(event: types.ErrorEvent):
    """Global error handler"""
    logger.error(f"❌ Bot error: {event.exception}", exc_info=True)
    
    try:
        error_msg = str(event.exception)
        if event.update.message:
            await event.update.message.answer(
                f"❌ <b>Error:</b> {error_msg[:200]}",
                parse_mode="HTML",
                reply_markup=UIComponents.main_menu()
            )
        elif event.update.callback_query:
            await event.update.callback_query.message.answer(
                f"❌ <b>Error:</b> {error_msg[:200]}",
                parse_mode="HTML",
                reply_markup=UIComponents.main_menu()
            )
            await event.update.callback_query.answer()
    except Exception as e:
        logger.error(f"❌ Error in error handler: {e}")

# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main function"""
    logger.info(f"🚀 Starting Bot Clone Proxy v{VERSION}")
    
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)  # WICHTIG: HTML statt Markdown
    )
    
    dp.include_router(router)
    
    logger.info("✅ Bot is running! Press Ctrl+C to stop.")
    
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        logger.info("🛑 Bot stopped")

if __name__ == "__main__":
    asyncio.run(main())
