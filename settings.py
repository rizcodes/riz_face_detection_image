#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 18:27:25 2021
Project setting information
"""
import os
import logging
import logging.config
from pathlib import Path


# Project directory paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROJ_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(ROOT_DIR, 'datastore_aiml', 'images',
                        'proj_binary_riz')
PROC_DIR = os.path.join(PROJ_DIR, 'proc')
LOG_DIR = os.path.join(PROJ_DIR, 'log')
MODEL_DIR = os.path.join(PROJ_DIR, 'models')

# Create proc and log directories
for folder in (PROC_DIR, LOG_DIR):
    os.makedirs(folder, exist_ok=True)


# Project directory details
project_dirs = {
    'base_dir': ROOT_DIR,
    'proj_dir': PROJ_DIR,
    'data_dir': DATA_DIR,
    'proc_dir': PROC_DIR,
    'model_dir': MODEL_DIR
    }

# Model pickle
conv_net_pkl = os.path.join(MODEL_DIR, 'ConvNet.pkl')

# Logging info
logging_definition = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'log_format': {
            'format': '[%(asctime)s] [%(filename)s:%(lineno)s] - %(levelname)s - %(message)s'
            },
        },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'log_format',
            'stream': 'ext://sys.stdout'
            },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'log_format',
            'filename': os.path.join(LOG_DIR, 'app.log'),
            'maxBytes': 1000*1024,
            'backupCount': 10
            }
        },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'DEBUG'
        }
    }

logging.config.dictConfig(logging_definition)
logger = logging.getLogger(__name__)
