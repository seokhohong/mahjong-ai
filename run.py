#!/usr/bin/env python3
"""
Launch script for Mahjong AI - Simple Jong
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.web.app import app

if __name__ == '__main__':
    print("🎴 Mahjong AI - Simple Jong")
    print("=" * 40)
    print("Starting the game server...")
    print("Open your web browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped. Thanks for playing!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1) 