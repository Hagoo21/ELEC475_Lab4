"""
Models package for ELEC475 Lab 4

This package contains the CLIP model implementation for image-text alignment.
"""

from .clip_model import CLIP, CLIPImageEncoder, CLIPTextEncoder, create_clip_model

__all__ = [
    'CLIP',
    'CLIPImageEncoder', 
    'CLIPTextEncoder',
    'create_clip_model'
]

