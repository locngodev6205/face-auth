"""Same realtime app with C to capture unlabelled expanded PAD crops.

Example: python app_capture.py --subject person01 --condition backlight
"""
import sys
from pathlib import Path

if __name__ == '__main__':
    if '--capture-dir' not in sys.argv and not any(a.startswith('--capture-dir=') for a in sys.argv):
        sys.argv.extend(['--capture-dir', str(Path(__file__).resolve().parent / 'pad_captures')])
    from app import main
    main()
