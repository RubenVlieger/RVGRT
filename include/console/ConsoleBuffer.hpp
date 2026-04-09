#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Message type for color coding in the console overlay
enum class ConsoleMsgType : uint8_t {
    System,    // Welcome, /help output — white
    Command,   // Echo of executed command — gray  
    Chat,      // Player chat messages — yellow
    Error      // Unknown command, etc. — red
};

struct ConsoleMessage {
    std::string text;
    ConsoleMsgType type;
    float timestamp;        // Seconds since game start (for fade-out)
    std::string senderName; // "Server", player name, or empty for system
};

// Circular buffer of console messages with scrolling support.
// Scroll offset: 0 = viewing the latest messages, increases as user scrolls up.
class ConsoleBuffer {
public:
    static constexpr size_t MAX_LINES = 200;
    static constexpr size_t VISIBLE_LINES = 20;

    ConsoleBuffer();
    ~ConsoleBuffer() = default;

    // Add a message. If user is at bottom (scrollOffset == 0), view stays at bottom.
    // If user scrolled up, view stays fixed (new messages accumulate below).
    void AddMessage(const std::string& text, ConsoleMsgType type,
                    const std::string& sender = "");

    // Scrolling
    void ScrollUp(size_t lines = 1);
    void ScrollDown(size_t lines = 1);
    void ScrollToBottom();

    // Returns up to `visibleCount` message pointers, ordered oldest-to-newest,
    // representing the window the user sees.
    std::vector<const ConsoleMessage*> GetVisibleLines(
        size_t visibleCount = VISIBLE_LINES) const;

    size_t GetScrollOffset() const { return _scrollOffset; }
    bool IsAtBottom() const { return _scrollOffset == 0; }
    size_t GetTotalMessageCount() const { return _messages.size(); }
    void Clear();

    // Advance the internal timestamp (call once per frame from GameConsole::Update)
    void AdvanceTime(float dt) { _currentTime += dt; }

private:
    std::vector<ConsoleMessage> _messages;
    size_t _scrollOffset;  // 0 = bottom, increases scrolling up through history
    float _currentTime;
};