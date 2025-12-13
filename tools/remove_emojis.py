"""
Remove ALL emojis from Python source files
Run this BEFORE building installer
"""
import re
from pathlib import Path
from typing import Dict

# Comprehensive emoji mapping
EMOJI_REPLACEMENTS: Dict[str, str] = {
    # Status emojis
    '✅': '[OK]',
    '❌': '[ERRO]',
    '⚠️': '[WARN]',
    'ℹ️': '[INFO]',
    '🔄': '[RETRY]',
    '📦': '[PACKAGE]',
    '🎮': '[GPU]',
    '🧹': '[CLEANUP]',
    '🎛️': '[CONTROL]',
    '📊': '[STATS]',

    # Audio/recording emojis
    '🎤': '[MIC]',
    '🔴': '[REC]',
    '⏹️': '[STOP]',
    '▶️': '[PLAY]',
    '⏸️': '[PAUSE]',
    '🎙️': '[MIC]',

    # Control emojis
    '✋': '[CANCELLED]',
    '🛑': '[STOP]',

    # Input emojis
    '🖱️': '[MOUSE]',
    '⌨️': '[KEYBOARD]',

    # Misc
    '🚀': '[START]',
    '💡': '[TIP]',
    '📝': '[NOTE]',
    '🔧': '[CONFIG]',
}

FILES_TO_PROCESS = [
    # Core source files
    'src/dictator/service.py',
    'src/dictator/tts_engine.py',
    'src/dictator/tray.py',
    'src/dictator/main.py',
    'src/dictator/health_check.py',
    'src/dictator/audio_processor.py',
    'src/dictator/voice/session_manager.py',
    'src/dictator/voice/llm_caller.py',
    'src/dictator/voice/vad_processor.py',

    # Installer files
    'installer/shared/validation.py',
    'installer/windows/dependency_checker.py',
    'installer/windows/config_wizard.py',
    'installer/windows/installer_core.py',
]

def remove_emojis_from_file(file_path: Path) -> tuple[int, list[str]]:
    """
    Remove all emojis from a file

    Returns:
        (count, found_emojis)
    """
    if not file_path.exists():
        return 0, []

    content = file_path.read_text(encoding='utf-8')
    original_content = content
    found_emojis = []

    # Replace known emojis
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        if emoji in content:
            found_emojis.append(emoji)
            content = content.replace(emoji, replacement)

    # Remove ANY remaining emojis using regex (replace with empty string)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )

    # Find remaining emojis BEFORE removal for warning
    remaining_before = emoji_pattern.findall(content)

    # Remove ALL remaining emojis
    content = emoji_pattern.sub('', content)

    if remaining_before:
        # Don't print emojis directly - will cause UnicodeEncodeError
        print(f"WARNING: {file_path} has {len(remaining_before)} unhandled emoji(s) - REMOVED")
        found_emojis.extend(['[OTHER]'] * len(remaining_before))

    if content != original_content:
        file_path.write_text(content, encoding='utf-8')
        return len(found_emojis), found_emojis

    return 0, []

def main():
    """Remove emojis from all source files"""
    base_dir = Path(__file__).parent.parent
    total_count = 0

    print("=" * 60)
    print("EMOJI REMOVAL SCRIPT")
    print("=" * 60)

    for file_rel in FILES_TO_PROCESS:
        file_path = base_dir / file_rel
        count, emojis = remove_emojis_from_file(file_path)

        if count > 0:
            print(f"\n{file_rel}")
            print(f"  Removed: {count} emojis")
            # Don't print actual emojis - will cause UnicodeEncodeError
            # print(f"  Found: {', '.join(emojis)}")
            total_count += count

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_count} emojis removed")
    print("=" * 60)

    if total_count > 0:
        print("\nNEXT STEPS:")
        print("1. Review changes with: git diff")
        print("2. Clear PyInstaller cache: rmdir /s /q build dist")
        print("3. Rebuild installer: cd installer/windows && python build_installer.py")

if __name__ == "__main__":
    main()
