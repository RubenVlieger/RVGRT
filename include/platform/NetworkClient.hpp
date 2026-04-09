#pragma once
#include <memory>
#include <vector>
#include <string>
#include <functional>

// Forward declaration
class Character;

// Callback type for incoming chat messages: (clientId, senderName, text)
using ChatCallback = std::function<void(int clientId, const std::string& sender, const std::string& text)>;

class NetworkClient {
public:
    virtual ~NetworkClient() = default;

    // Connects asynchronously to the specified WebSocket URL (e.g., "ws://127.0.0.1:8000/ws")
    virtual void Connect(const std::string& url) = 0;

    // Disconnects from the server
    virtual void Disconnect() = 0;

    // Serializes the local character's matrices to JSON and sends them
    virtual void SendState(const Character& localCharacter) = 0;

    // Sends a chat message to the server
    virtual void SendChat(const std::string& sender, const std::string& text) = 0;

    // Called once per frame on the main thread.
    // Drains any newly received states and directly updates the destination array
    // (State::state.otherCharacters).
    virtual void PollUpdates(std::vector<Character>& otherCharacters) = 0;

    // Set the callback for incoming chat messages
    virtual void SetChatCallback(ChatCallback callback) = 0;

    // Factory method to allocate the OS-specific implementation
    static std::unique_ptr<NetworkClient> Create();
};
