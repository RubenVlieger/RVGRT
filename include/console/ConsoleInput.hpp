#pragma once

#include <string>
#include <vector>

// Text input state machine for the console.
// Supports cursor movement, insertion, deletion, and command history.
class ConsoleInput {
public:
    static constexpr size_t MAX_INPUT_LENGTH = 256;
    static constexpr size_t MAX_HISTORY = 50;

    ConsoleInput();
    ~ConsoleInput() = default;

    // Insert a printable character at the current cursor position
    void InsertChar(char c);

    // Editing
    void Backspace();     // Delete character before cursor
    void Delete();        // Delete character at cursor
    void Clear();

    // Cursor movement
    void MoveCursorLeft();
    void MoveCursorRight();
    void MoveCursorHome();
    void MoveCursorEnd();

    // Command history navigation
    void HistoryUp();
    void HistoryDown();
    bool IsBrowsingHistory() const { return _historyIndex != SIZE_MAX; }

    // Save current input to history (called on Enter/submit)
    void PushHistory();

    // Accessors
    const std::string& GetText() const { return _input; }
    size_t GetCursorPos() const { return _cursorPos; }
    bool IsEmpty() const { return _input.empty(); }

    // Prefix ('/' for command mode, 0 for chat mode)
    void SetPrefix(char prefix) { _prefix = prefix; }
    char GetPrefix() const { return _prefix; }
    std::string GetFullText() const;   // prefix + input

    // Cursor display position (accounts for prefix length)
    size_t GetDisplayCursorPos() const;

private:
    std::string _input;
    size_t _cursorPos;
    char _prefix;                  // 0 = no prefix, '/' = command prefix

    // Command history (most recent last)
    std::vector<std::string> _history;
    size_t _historyIndex;          // Current position in history; SIZE_MAX = "live" input
    std::string _savedInput;       // Input being edited before history navigation

    void RestoreFromHistory();
};