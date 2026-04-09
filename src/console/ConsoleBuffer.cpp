#include "console/ConsoleBuffer.hpp"
#include <algorithm>

ConsoleBuffer::ConsoleBuffer()
    : _scrollOffset(0)
    , _currentTime(0.0f) {
}

void ConsoleBuffer::AddMessage(const std::string& text, ConsoleMsgType type,
                               const std::string& sender) {
    ConsoleMessage msg;
    msg.text = text;
    msg.type = type;
    msg.timestamp = _currentTime;
    msg.senderName = sender;

    _messages.push_back(std::move(msg));

    // Keep buffer bounded — erase oldest when over capacity
    if (_messages.size() > MAX_LINES) {
        _messages.erase(_messages.begin());
        // Adjust scroll offset: we removed one message from the front,
        // so if user was scrolled up, they should still see the same content
        if (_scrollOffset > 0) {
            _scrollOffset--;
        }
    }

    // If user is at the bottom, they stay at the bottom (offset stays 0).
    // If user scrolled up, they stay at the same historical position.
    // We do NOT auto-scroll to bottom when the user is reading history.
}

void ConsoleBuffer::ScrollUp(size_t lines) {
    // Maximum scroll: we can scroll until the topmost visible line
    // aligns with the first message in the buffer.
    size_t maxMessages = _messages.size();
    if (maxMessages <= VISIBLE_LINES) {
        // Not enough messages to scroll
        return;
    }
    size_t maxOffset = maxMessages - VISIBLE_LINES;
    size_t newOffset = _scrollOffset + lines;
    _scrollOffset = std::min(newOffset, maxOffset);
}

void ConsoleBuffer::ScrollDown(size_t lines) {
    if (_scrollOffset > lines) {
        _scrollOffset -= lines;
    } else {
        _scrollOffset = 0;
    }
}

void ConsoleBuffer::ScrollToBottom() {
    _scrollOffset = 0;
}

std::vector<const ConsoleMessage*> ConsoleBuffer::GetVisibleLines(
    size_t visibleCount) const {

    std::vector<const ConsoleMessage*> result;

    if (_messages.empty()) {
        return result;
    }

    size_t totalMessages = _messages.size();

    // Determine how many messages we can actually show
    size_t count = std::min(visibleCount, totalMessages);

    // The bottommost visible message index (counting from 0 at front)
    // scrollOffset=0 means we see the last `count` messages.
    // scrollOffset=N means we see messages ending N positions earlier.
    size_t endIdx = totalMessages - _scrollOffset;   // exclusive upper bound
    size_t startIdx = endIdx - count;                  // inclusive lower bound

    for (size_t i = 0; i < count; ++i) {
        result.push_back(&_messages[startIdx + i]);
    }

    return result;
}

void ConsoleBuffer::Clear() {
    _messages.clear();
    _scrollOffset = 0;
}