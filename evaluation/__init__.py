"""
Image Editing Evaluation Tools

This package provides comprehensive evaluation tools for image editing tasks,
based on AnyBench's evaluation framework and integrated with NeoAnalogist.

Modules:
- emu_edit_evaluator: Full-featured evaluator for Emu-Edit dataset
- simple_evaluator: Lightweight evaluator for single image pairs
- test_evaluation: Test suite for validation

Usage:
    from evaluation.simple_evaluator import SimpleImageEditEvaluator
    
    evaluator = SimpleImageEditEvaluator()
    results = evaluator.evaluate_with_analysis(
        original_path="original.jpg",
        edited_path="edited.jpg", 
        caption="Change the cat to a dog"
    )
"""

from .simple_evaluator import SimpleImageEditEvaluator
from .emu_edit_evaluator import EmuEditEvaluator

__version__ = "1.0.0"
__author__ = "NeoAnalogist Team"

__all__ = [
    "SimpleImageEditEvaluator",
    "EmuEditEvaluator"
]
