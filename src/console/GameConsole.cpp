#include "console/GameConsole.hpp"
#include "console/RegisterCommands.hpp"

#ifdef _DEBUG
#include <cstdio>
#define CONSOLE_LOG(fmt, ...) fprintf(stderr, "[GameConsole] " fmt "\n", ##__VA_ARGS__)
#else
#define CONSOLE_LOG(fmt, ...) ((void)0)
#endif

GameConsole::GameConsole()
    : _isOpen(false)
    , _openPrefix(0)
    , _cursorBlinkTime(0.0f)
    , _cursorVisible(true)
    , _playerName("Player")
    , _clientId(-1)
    , _chatCallback(nullptr) {
}

void GameConsole::Initialize() {
    RegisterAllCommands(*this);

    _buffer.AddMessage("Welcome to RVGRT!", ConsoleMsgType::System);
    _buffer.AddMessage("Type /help to see all possible commands", ConsoleMsgType::System);

    CONSOLE_LOG("Initialized with %d commands", (int)_registry.GetAllCommands().size());
}

void GameConsole::Open(char prefix) {
    if (_isOpen) return;  // Already open

    _isOpen = true;
    _openPrefix = prefix;
    _input.Clear();
    _input.SetPrefix(prefix);
    // When opening with '/', the prefix '/' is prepended automatically
    // by GetFullText(). The user types the command name after it.
    _cursorBlinkTime = 0.0f;
    _cursorVisible = true;

    CONSOLE_LOG("Opened with prefix '%c'", prefix ? prefix : 'T');
}

void GameConsole::Close() {
    _isOpen = false;
    _openPrefix = 0;
    _input.SetPrefix(0);
    _input.Clear();
}

void GameConsole::Toggle() {
    if (_isOpen) {
        Close();
    } else {
        Open(0);
    }
}

void GameConsole::Update(float deltaTime) {
    _buffer.AdvanceTime(deltaTime);

    if (!_isOpen) return;

    // Blink the cursor every 500ms
    _cursorBlinkTime += deltaTime;
    if (_cursorBlinkTime >= 0.5f) {
        _cursorBlinkTime -= 0.5f;
        _cursorVisible = !_cursorVisible;
    }
}

void GameConsole::OnCharInput(char c) {
    if (!_isOpen) return;
    _input.InsertChar(c);
}

void GameConsole::OnSpecialKey(SpecialKey key) {
    if (!_isOpen) return;

    switch (key) {
        case SpecialKey::Enter:
            SubmitInput();
            break;

        case SpecialKey::Backspace:
            _input.Backspace();
            break;

        case SpecialKey::Delete:
            _input.Delete();
            break;

        case SpecialKey::Escape:
            Close();
            break;

        case SpecialKey::ArrowUp:
            // If input is empty and we're at the bottom, browse history.
            // Otherwise, scroll the buffer.
            if (_input.IsEmpty()) {
                _input.HistoryUp();
            } else {
                _buffer.ScrollUp();
            }
            break;

        case SpecialKey::ArrowDown:
            if (_input.IsBrowsingHistory()) {
                _input.HistoryDown();
            } else {
                _buffer.ScrollDown();
            }
            break;

        case SpecialKey::Home:
            _input.MoveCursorHome();
            break;

        case SpecialKey::End:
            _input.MoveCursorEnd();
            break;
    }
}

void GameConsole::OnChatReceived(int clientId, const std::string& sender,
                                 const std::string& text) {
    // Don't echo our own messages back
    if (clientId == _clientId) return;

    std::string displayText;
    if (!sender.empty()) {
        displayText = "<" + sender + "> " + text;
    } else {
        displayText = text;
    }

    _buffer.AddMessage(displayText, ConsoleMsgType::Chat, sender);
    CONSOLE_LOG("Chat from client %d (%s): %s", clientId, sender.c_str(), text.c_str());
}

void GameConsole::GetVisibleMessages(
    std::vector<const ConsoleMessage*>& outMessages) const {
    outMessages = _buffer.GetVisibleLines();
}

std::string GameConsole::GetInputDisplayText() const {
    std::string text = "> ";
    text += _input.GetFullText();
    if (_cursorVisible && _isOpen) {
        text += "|";
    }
    return text;
}

size_t GameConsole::GetDisplayCursorPos() const {
    // "> " = 2 chars prefix, plus 1 char if there's a / prefix
    return 2 + _input.GetDisplayCursorPos();
}

void GameConsole::SetPlayerName(const std::string& name) {
    _playerName = name;
    CONSOLE_LOG("Player name set to: %s", name.c_str());
}

void GameConsole::SubmitInput() {
    std::string fullText = _input.GetFullText();

    if (fullText.empty() || (fullText.size() == 1 && fullText[0] == '/')) {
        // Empty input or just "/" — close the console
        Close();
        return;
    }

    // Push to history before processing (so we can recall it later)
    _input.PushHistory();

    if (fullText[0] == '/') {
        // Command — echo it and execute
        _buffer.AddMessage(fullText, ConsoleMsgType::Command);

        bool executed = _registry.Execute(fullText, *this);
        if (!executed) {
            // Extract just the command name for the error message
            std::string cmdName;
            size_t spacePos = fullText.find(' ');
            if (spacePos != std::string::npos) {
                cmdName = fullText.substr(1, spacePos - 1);
            } else {
                cmdName = fullText.substr(1);
            }
            _buffer.AddMessage("Unknown command '" + cmdName +
                             "'. Type /help to see all possible commands",
                             ConsoleMsgType::Error);
        }
    } else {
        // Regular chat message
        std::string displayText = "<" + _playerName + "> " + fullText;
        _buffer.AddMessage(displayText, ConsoleMsgType::Chat, _playerName);

        if (_chatCallback) {
            _chatCallback(_playerName, fullText);
        }
    }

    // Clear input and close the console (Minecraft behavior: Enter closes chat)
    Close();
}