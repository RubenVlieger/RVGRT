#pragma once

#include "console/ConsoleBuffer.hpp"
#include "console/ConsoleInput.hpp"
#include "console/CommandRegistry.hpp"
#include <functional>
#include <string>
#include <vector>

// Special keys for console input (platform-agnostic)
enum class SpecialKey {
    Enter,
    Backspace,
    Delete,
    Escape,
    ArrowUp,
    ArrowDown,
    Home,
    End
};

// Callback type for sending chat messages to the server: (senderName, text)
using ChatSendCallback = std::function<void(const std::string& sender, const std::string& text)>;

// Main coordinator for the in-game console.
// Owns the message buffer, input state, and command registry.
// Routes keyboard input, manages open/close state, and
// dispatches chat/command submissions.
class GameConsole {
public:
    GameConsole();
    ~GameConsole() = default;

    // Call once after construction, before first use
    void Initialize();

    // --- Opening / closing ---
    // prefix=0   → chat mode (T key): input starts empty
    // prefix='/' → command mode (/ key): input starts with "/" pre-filled
    void Open(char prefix = 0);
    void Close();
    void Toggle();
    bool IsOpen() const { return _isOpen; }

    // --- Per-frame update ---
    // Call from game loop. Handles cursor blink timer.
    void Update(float deltaTime);

    // --- Input events ---
    // Route printable characters here when console is open
    void OnCharInput(char c);
    // Route special keys here when console is open
    void OnSpecialKey(SpecialKey key);

    // --- Network ---
    // Called when a chat message arrives from the server
    void OnChatReceived(int clientId, const std::string& sender,
                        const std::string& text);
    // Set the callback that sends text to the server
    void SetChatSendCallback(ChatSendCallback cb) { _chatCallback = cb; }
    // Get the chat send callback (used by /me command)
    ChatSendCallback GetChatSendCallback() const { return _chatCallback; }

    // --- Rendering data ---
    // Fills `outMessages` with pointers to the currently visible lines
    void GetVisibleMessages(std::vector<const ConsoleMessage*>& outMessages) const;
    // Returns the input line with prompt, prefix, text, and blinking cursor
    std::string GetInputDisplayText() const;
    // Returns the cursor column for rendering the blinking cursor
    size_t GetDisplayCursorPos() const;
    bool GetCursorVisible() const { return _cursorVisible; }

    // --- Component accessors ---
    ConsoleBuffer& GetBuffer() { return _buffer; }
    ConsoleInput& GetInput() { return _input; }
    CommandRegistry& GetRegistry() { return _registry; }

    // --- Player identity ---
    void SetPlayerName(const std::string& name);
    const std::string& GetPlayerName() const { return _playerName; }
    void SetPlayerClientId(int id) { _clientId = id; }
    int GetPlayerClientId() const { return _clientId; }

private:
    ConsoleBuffer _buffer;
    ConsoleInput _input;
    CommandRegistry _registry;

    bool _isOpen;
    char _openPrefix;
    float _cursorBlinkTime;
    bool _cursorVisible;

    std::string _playerName;
    int _clientId;

    ChatSendCallback _chatCallback;

    // Process the current input line (Enter key)
    void SubmitInput();
};