#include "console/ConsoleInput.hpp"
#include <algorithm>

ConsoleInput::ConsoleInput()
    : _cursorPos(0)
    , _prefix(0)
    , _historyIndex(SIZE_MAX)   // SIZE_MAX means "not browsing history"
    , _savedInput("") {
}

void ConsoleInput::InsertChar(char c) {
    // Only accept printable ASCII
    if (c < 32 || c > 126) {
        return;
    }
    if (_input.length() >= MAX_INPUT_LENGTH) {
        return;
    }
    _input.insert(_cursorPos, 1, c);
    _cursorPos++;
}

void ConsoleInput::Backspace() {
    if (_cursorPos > 0 && !_input.empty()) {
        _input.erase(_cursorPos - 1, 1);
        _cursorPos--;
    }
}

void ConsoleInput::Delete() {
    if (_cursorPos < _input.length()) {
        _input.erase(_cursorPos, 1);
    }
}

void ConsoleInput::Clear() {
    _input.clear();
    _cursorPos = 0;
    _historyIndex = SIZE_MAX;
    _savedInput.clear();
}

void ConsoleInput::MoveCursorLeft() {
    if (_cursorPos > 0) {
        _cursorPos--;
    }
}

void ConsoleInput::MoveCursorRight() {
    if (_cursorPos < _input.length()) {
        _cursorPos++;
    }
}

void ConsoleInput::MoveCursorHome() {
    _cursorPos = 0;
}

void ConsoleInput::MoveCursorEnd() {
    _cursorPos = _input.length();
}

void ConsoleInput::HistoryUp() {
    if (_history.empty()) return;

    // Save current input before entering history for the first time
    if (_historyIndex == SIZE_MAX) {
        _savedInput = _input;
        _historyIndex = _history.size() - 1;
    } else if (_historyIndex > 0) {
        _historyIndex--;
    }
    // If historyIndex == 0, we're already at the oldest entry; stay there

    RestoreFromHistory();
}

void ConsoleInput::HistoryDown() {
    if (_historyIndex == SIZE_MAX) return;  // Not in history

    if (_historyIndex < _history.size() - 1) {
        _historyIndex++;
        RestoreFromHistory();
    } else {
        // Exit history — restore the input the user was typing
        _historyIndex = SIZE_MAX;
        _input = _savedInput;
        _cursorPos = _input.length();
    }
}

void ConsoleInput::PushHistory() {
    if (_input.empty()) return;

    // Don't add duplicates of the most recent entry
    if (!_history.empty() && _history.back() == _input) {
        _historyIndex = SIZE_MAX;
        _savedInput.clear();
        return;
    }

    _history.push_back(_input);

    // Keep history bounded
    if (_history.size() > MAX_HISTORY) {
        _history.erase(_history.begin());
    }

    _historyIndex = SIZE_MAX;
    _savedInput.clear();
}

std::string ConsoleInput::GetFullText() const {
    if (_prefix != 0) {
        return std::string(1, _prefix) + _input;
    }
    return _input;
}

size_t ConsoleInput::GetDisplayCursorPos() const {
    // The display position accounts for the "> " prompt and prefix
    // "> " = 2 chars, prefix = 1 char if nonzero
    return _cursorPos + (_prefix != 0 ? 1 : 0);
}

void ConsoleInput::RestoreFromHistory() {
    if (_historyIndex < _history.size()) {
        _input = _history[_historyIndex];
        _cursorPos = _input.length();
    }
}